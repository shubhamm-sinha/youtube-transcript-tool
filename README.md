---
title: YouTube Transcript Tool
emoji: 📺
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# YouTube Transcript Tool (free, captions first, Whisper fallback)

Paste a YouTube URL, get a transcript. Captions are checked first (fast,
2-5 s); when a video has no captions, the audio is transcribed with Whisper,
so it still returns a transcript. Translation to the target language included.

## How it works

1. `youtube-transcript-api` checks for captions: manual first, then
   auto-generated, in the target language if available. YouTube's own
   translated captions are used when the caption language differs.
2. If no captions exist, `yt-dlp` downloads audio only (no video),
   `ffmpeg` converts to 16 kHz mono WAV.
3. Whisper transcribes the audio in its original language, with timestamps
   per segment. Whisper auto-detects the language (99 languages: Hindi,
   Tamil, Bengali, etc).
4. The transcript text is translated to the target language in 4000-char chunks,
   timestamps are kept separate so they never break.
5. Results are cached in SQLite by video ID, so the same URL is instant the second time.
   Every result says where it came from: `captions` or `whisper`.

## Transcription engine (two modes, both free)

- Local (default, no key needed): `faster-whisper`, model `base` by default.
  `small` is more accurate, `medium` needs a GPU to be fast.
  Needs about 2 GB RAM for `small`. A 1-minute video takes roughly 30-60 s on CPU.
- Fast (optional): set `GROQ_API_KEY` and `WHISPER_BACKEND=groq`.
  Groq's free Whisper API tier runs the same model on GPU, about 20-30 s for a 10-min video.
  Local faster-whisper stays as the fallback if Groq fails or no key is set.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, paste a URL, pick a target language, watch progress.

Or use the API directly:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://www.youtube.com/watch?v=S4OYqw9KZOU", "target_lang": "en"}'
# -> {"job_id": "...", "status": "queued"}
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl http://127.0.0.1:8000/api/jobs/<job_id>/download?format=txt   # or srt
```

## Free hosting

- API on Render free tier (512 MB RAM) works only with `WHISPER_BACKEND=groq`,
  because local Whisper needs more RAM than the free tier gives.
- Fully self-hosted free: Oracle Always Free VM (4 CPU, 24 GB RAM) runs
  local faster-whisper for $0. Frontend on Vercel or Netlify, free.
- Small demo: Hugging Face Spaces free CPU.

## Files

- `app.py` - FastAPI app, job queue, cache, download endpoints, serves the frontend
- `transcribe.py` - URL parsing, audio download, Whisper, translation, SRT/TXT output
- `static/index.html` - frontend: paste URL, target language, progress, transcript, copy/download
- `requirements.txt`, `.env.example`

## Limits (honest ones)

- Public videos only. Private, age-restricted, or region-blocked videos fail at download.
- Translation uses the free Google Translate endpoint via `deep-translator`,
  with MyMemory as fallback; both are unofficial and can rate-limit on
  shared or datacenter IPs. Self-host LibreTranslate for a fully owned fallback.
- Long videos take a while: transcription is a background job, the frontend polls
  for status instead of waiting on one request, so nothing times out.

## Tested 2026-11-20 in this VM (what was proven, what was blocked)

Proven, on Shubham's own Day 14 CloudWatch Short audio (53 s, local file):
- `faster-whisper` base, CPU, transcribed 16 segments in about 10 s,
  detected language `en`, text matches the episode script almost verbatim
  (hook "3am, your server breaks, no one knows until users complain",
  myth, CTA, Day 15 preview). This is the Whisper fallback path.
- SRT and TXT output generation works from those segments.
- Sample outputs saved in this folder: `sample-day14-cloudwatch.txt` / `.srt`.

Blocked by this VM's shared egress, not by the code:
- Caption check: `youtube-transcript-api` calls YouTube's player API,
  which this egress blocks (proxy disconnect / 403). The code treats any
  caption failure as "no captions" and falls back to Whisper, so this
  cannot break a job, it just makes the fast path untestable here.
- `yt-dlp` download: same player API 403 (watch page HTML via curl works,
  `youtubei/v1/player` returns 403).
- Translation: Google free endpoint and MyMemory both returned 429
  (rate-limited) from this shared IP. Translation code keeps Google first,
  MyMemory second, and a failed translation never loses the transcript:
  the job returns the original-language transcript instead.
- Two VM-only fixes are in the code: `HF_HUB_DISABLE_XET=1` (HF's Xet
  download path is blocked here, plain downloads work) and WAV decoding
  in pure Python instead of faster-whisper's PyAV path (av 19 removed
  the `metadata_errors` kwarg it passes).
