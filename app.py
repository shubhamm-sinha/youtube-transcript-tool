"""FastAPI front end for the Whisper-first YouTube transcript tool.

Jobs run in a background thread so long videos never time out;
the frontend polls GET /api/jobs/{job_id}. Results are cached in
SQLite by (video_id, target_lang).
"""
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transcribe import extract_video_id, run_transcript

BASE = Path(__file__).parent
DB = BASE / "cache.db"
STATIC = BASE / "static"

app = FastAPI(title="YouTube Transcript Tool")
jobs: dict[str, dict] = {}


def _db():
    conn = sqlite3.connect(DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS transcripts"
        " (video_id TEXT, target_lang TEXT, result TEXT,"
        " PRIMARY KEY (video_id, target_lang))"
    )
    return conn


def _cache_get(video_id: str, target_lang: str):
    conn = _db()
    row = conn.execute(
        "SELECT result FROM transcripts WHERE video_id=? AND target_lang=?",
        (video_id, target_lang)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _cache_put(video_id: str, target_lang: str, result: dict):
    conn = _db()
    conn.execute(
        "INSERT OR REPLACE INTO transcripts VALUES (?, ?, ?)",
        (video_id, target_lang, json.dumps(result)))
    conn.commit()
    conn.close()


class JobRequest(BaseModel):
    url: str
    target_lang: str = "en"


def _run_job(job_id: str, url: str, target_lang: str):
    try:
        video_id = extract_video_id(url)
        cached = _cache_get(video_id, target_lang)
        if cached:
            jobs[job_id] = {"status": "done", "cached": True, **cached}
            return
        jobs[job_id]["status"] = "checking captions"
        result = run_transcript(url, target_lang,
                                on_status=lambda s: jobs[job_id].__setitem__("status", s))
        _cache_put(video_id, target_lang, result)
        jobs[job_id] = {"status": "done", "cached": False, **result}
    except Exception as exc:  # noqa: BLE001 - surface the failure to the user
        jobs[job_id] = {"status": "error", "error": str(exc)}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    try:
        extract_video_id(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued"}
    threading.Thread(target=_run_job, args=(job_id, req.url, req.target_lang),
                     daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str, format: str = "txt"):  # noqa: A002
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="transcript not ready")
    text = job["srt"] if format == "srt" else job["txt"]
    return PlainTextResponse(text, media_type="text/plain",
                             headers={"Content-Disposition":
                                      f'attachment; filename="transcript.{format}"'})


if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
