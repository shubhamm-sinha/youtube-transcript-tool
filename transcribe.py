"""Whisper-first YouTube transcript core.

Captions are never used as a source. Every public video goes through:
YouTube URL -> yt-dlp audio only -> ffmpeg 16kHz mono WAV
-> Whisper (local faster-whisper, or Groq free API if configured)
-> transcript in original language -> translate to target language.
"""
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path

# This VM's egress blocks Hugging Face's Xet download path; plain HTTP
# downloads work. Only affects model download here, harmless elsewhere.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import yt_dlp

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "local")  # local | groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_VIDEO_ID_RE = re.compile(
    r"(?:v=|/shorts/|/embed/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str:
    """Parse a video ID from watch, youtu.be, Shorts, embed and live URLs."""
    m = _VIDEO_ID_RE.search(url or "")
    if not m:
        raise ValueError("could not find a YouTube video ID in that URL")
    return m.group(1)


def download_audio(video_id: str, workdir: Path) -> Path:
    """Download audio only, convert to 16kHz mono WAV for Whisper."""
    raw = workdir / "audio.%(ext)s"
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(raw),
        "quiet": True,
        "noplaylist": True,
        # This VM egresses through a proxy with its own CA; yt-dlp uses
        # certifi, not SSL_CERT_FILE, so point it at the bundle when present.
        # On a normal machine / host this is absent and yt-dlp uses its default.
        **({"cafile": os.environ["SSL_CERT_FILE"]}
           if os.environ.get("SSL_CERT_FILE")
           and os.path.exists(os.environ["SSL_CERT_FILE"]) else {}),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as exc:
            msg = str(exc)
            if "403" in msg or "Failed to extract any player response" in msg:
                raise RuntimeError(
                    "YouTube blocked the download from this server "
                    "(its player API returns 403 for datacenter/proxy IPs). "
                    "On your own machine or an Oracle Always Free VM this "
                    "usually works; if it still fails, add your browser "
                    "cookies with yt-dlp --cookies-from-browser chrome.") from exc
            raise
    wav = workdir / "audio.wav"
    if not wav.exists():
        found = sorted(workdir.glob("audio.*"))
        if not found:
            raise RuntimeError("yt-dlp finished but no audio file was produced")
        wav = found[0]
    return wav


def try_captions(video_id: str, target_lang: str):
    """Caption layer (fast path, 2-5 s). Returns a result dict or None.

    Prefers a transcript already in the target language, then a manual
    transcript in any language, then auto-generated. Translation uses
    YouTube's own translated captions when available, else the free
    translator chain. Any failure returns None so the caller falls
    back to Whisper; captions are a fast path, never a hard dependency.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        listing = YouTubeTranscriptApi().list(video_id)
    except Exception:
        return None
    available = list(listing)
    if not available:
        return None
    by_code = {t.language_code: t for t in available}
    chosen, translated = None, False
    if target_lang in by_code:
        chosen = by_code[target_lang]
    else:
        manual = [t for t in available if not t.is_generated]
        chosen = manual[0] if manual else available[0]
    try:
        fetched = chosen.fetch()
        source_lang = chosen.language_code
        if source_lang != target_lang:
            try:
                fetched = chosen.translate(target_lang).fetch()
                translated = True
            except Exception:
                pass
        segments = [{
            "start": round(s.start, 2),
            "end": round(s.start + s.duration, 2),
            "text": s.text.strip(),
        } for s in fetched if s.text.strip()]
    except Exception:
        return None
    if not segments:
        return None
    if source_lang != target_lang and not translated:
        try:
            segments, translated = translate_segments(segments, source_lang, target_lang)
        except Exception:
            pass
    return {
        "video_id": video_id,
        "source": "captions",
        "source_lang": source_lang,
        "target_lang": target_lang if translated else source_lang,
        "translated": translated,
        "segments": segments,
        "txt": to_txt(segments),
        "srt": to_srt(segments),
    }


def _load_wav_array(wav: Path):
    """Decode a 16kHz mono WAV to float32 numpy, avoiding faster-whisper's
    PyAV decode path (av 19 removed the metadata_errors kwarg it passes)."""
    import numpy as np

    with wave.open(str(wav), "rb") as w:
        n_channels, sampwidth, rate, n_frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sampwidth}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    if rate != 16000:
        raise ValueError(f"expected 16kHz WAV, got {rate}Hz")
    return audio


def transcribe_local(wav: Path, model_name: str = WHISPER_MODEL):
    """Local faster-whisper transcription. Returns (segments, detected_lang)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    audio = _load_wav_array(wav)
    segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)
    out = [{
        "start": round(s.start, 2),
        "end": round(s.end, 2),
        "text": s.text.strip(),
    } for s in segments if s.text.strip()]
    return out, info.language


def transcribe_groq(wav: Path):
    """Groq free Whisper API tier. Same model, GPU speed. Needs GROQ_API_KEY."""
    import httpx

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    with open(wav, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "whisper-large-v3-turbo", "response_format": "verbose_json",
                "timestamp_granularities[]": "segment"}
        r = httpx.post("https://api.groq.com/openai/v1/audio/transcriptions",
                       headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                       files=files, data=data, timeout=600)
    r.raise_for_status()
    payload = r.json()
    out = [{
        "start": round(s["start"], 2),
        "end": round(s["end"], 2),
        "text": s["text"].strip(),
    } for s in payload.get("segments", []) if s["text"].strip()]
    return out, payload.get("language", "unknown")


def _translate_piece(text: str, source_lang: str, target_lang: str) -> str:
    """One chunk, Google first, MyMemory as fallback.

    Google's free endpoint rate-limits shared/proxy IPs (429 seen 2026-11-20
    in this VM), so a failure must not kill the whole transcript.
    MyMemory needs locale codes like en-GB / hi-IN, not bare en / hi.
    """
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        locale = {"en": "en-GB", "hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN",
                  "mr": "mr-IN", "es": "es-ES", "fr": "fr-FR", "de": "de-DE"}
        return MyMemoryTranslator(
            source=locale.get(source_lang, source_lang),
            target=locale.get(target_lang, target_lang)).translate(text)


def translate_segments(segments, source_lang: str, target_lang: str):
    """Translate segment text only, timestamps untouched. Chunked at 4000 chars."""
    if not target_lang or target_lang == source_lang:
        return segments, False
    # Join with a marker that survives translation, then split back per segment.
    marker = "\n@@@\n"
    joined = marker.join(s["text"] for s in segments)
    # Chunk the joined text so no request exceeds 4000 chars, on marker boundaries.
    pieces, current = [], ""
    for text in [s["text"] for s in segments]:
        piece = text if not current else marker + text
        if len(current) + len(piece) > 4000 and current:
            pieces.append(current)
            current = text
        else:
            current += piece
    if current:
        pieces.append(current)
    out_texts = []
    for piece in pieces:
        out_texts.append(_translate_piece(piece, source_lang, target_lang))
    translated = marker.join(out_texts).split(marker)
    if len(translated) != len(segments):
        # Translation merged or split a marker; fall back to per-segment translate.
        translated = [_translate_piece(s["text"], source_lang, target_lang) for s in segments]
    return [{**s, "text": t.strip()} for s, t in zip(segments, translated)], True


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_srt(segments) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(s['start'])},000 --> {fmt_time(s['end'])},000")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def to_txt(segments) -> str:
    return "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in segments)


def run_transcript(url: str, target_lang: str = "en", model: str = WHISPER_MODEL,
                   on_status=None):
    """Hybrid pipeline, as decided 2026-11-20 11:40 IST: captions first,
    Whisper fallback when no captions exist. Returns dict with segments,
    languages, text outputs, and 'source' = 'captions' or 'whisper'."""
    video_id = extract_video_id(url)
    if on_status:
        on_status("checking captions")
    captioned = try_captions(video_id, target_lang)
    if captioned:
        return captioned
    if on_status:
        on_status("no captions, transcribing audio with Whisper")
    with tempfile.TemporaryDirectory(prefix="yt-transcript-") as tmp:
        workdir = Path(tmp)
        wav = download_audio(video_id, workdir)
        if WHISPER_BACKEND == "groq":
            try:
                segments, detected = transcribe_groq(wav)
            except Exception:
                segments, detected = transcribe_local(wav, model)
        else:
            segments, detected = transcribe_local(wav, model)
    if not segments:
        raise RuntimeError("Whisper returned no speech segments")
    try:
        translated, did_translate = translate_segments(segments, detected, target_lang)
    except Exception:
        # Translation is best-effort; a rate-limited translator must not
        # lose a transcript that Whisper already produced.
        translated, did_translate = segments, False
    shown = translated if did_translate else segments
    return {
        "video_id": video_id,
        "source": "whisper",
        "source_lang": detected,
        "target_lang": target_lang if did_translate else detected,
        "translated": did_translate,
        "segments": shown,
        "txt": to_txt(shown),
        "srt": to_srt(shown),
    }
