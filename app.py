"""
VideoExtracterFromURL — Backend
================================
FastAPI server with yt-dlp for universal video downloading.
Supports 1000+ platforms: YouTube, Wistia, Vimeo, Twitter/X, TikTok, etc.

Run:
    python app.py
    Then open: http://localhost:8000
"""

import glob
import json
import os
import queue
import threading
import uuid
from pathlib import Path

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path("downloads")
DOWNLOADS_DIR.mkdir(exist_ok=True)

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="VideoExtracterFromURL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-Memory Job Store ─────────────────────────────────────────────────────
# Structure: { job_id: { queue, status, filename, filepath, error } }

jobs: dict = {}
jobs_lock = threading.Lock()


# ─── Models ──────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: str = "best"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    """Serve the frontend HTML page."""
    return FileResponse("templates/index.html")


@app.get("/api/info")
def get_video_info(url: str):
    """
    Fetch video metadata (title, thumbnail, formats) without downloading.
    Called when user clicks Analyze.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Build format list — only include formats with video
            formats = []
            for f in info.get("formats", []):
                if f.get("vcodec") and f.get("vcodec") != "none":
                    size = f.get("filesize") or f.get("filesize_approx")
                    formats.append({
                        "format_id": f.get("format_id"),
                        "ext": f.get("ext", "mp4"),
                        "resolution": f.get("resolution") or (
                            f"{f.get('height', '?')}p" if f.get("height") else "unknown"
                        ),
                        "filesize": size,
                        "note": f.get("format_note", ""),
                    })

            # Return highest quality formats (last = best in yt-dlp ordering)
            top_formats = formats[-8:] if len(formats) > 8 else formats
            top_formats.reverse()

            duration = info.get("duration", 0)
            minutes = int(duration // 60) if duration else 0
            seconds = int(duration % 60) if duration else 0

            return {
                "title": info.get("title", "Unknown Title"),
                "thumbnail": info.get("thumbnail"),
                "duration": f"{minutes}:{seconds:02d}" if duration else "Unknown",
                "uploader": info.get("uploader") or info.get("channel", "Unknown"),
                "platform": info.get("extractor_key", "Unknown"),
                "formats": top_formats,
            }

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=str(e).replace("ERROR: ", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download")
def start_download(req: DownloadRequest):
    """
    Kick off a background download job.
    Returns a job_id used to track progress via SSE.
    """
    job_id = str(uuid.uuid4())
    job_queue: queue.Queue = queue.Queue()

    with jobs_lock:
        jobs[job_id] = {
            "queue": job_queue,
            "status": "starting",
            "filename": None,
            "filepath": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_download,
        args=(job_id, req.url, req.format_id),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
def stream_progress(job_id: str):
    """
    Server-Sent Events (SSE) stream for real-time download progress.
    Frontend connects here after receiving job_id.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_generator():
        job_queue = jobs[job_id]["queue"]
        while True:
            try:
                event = job_queue.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("finished", "error"):
                    break
            except queue.Empty:
                # Keepalive ping so the SSE connection doesn't time out
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/file/{job_id}")
def download_file(job_id: str):
    """Serve the completed download file to the browser."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    filepath = job.get("filepath")

    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not ready or not found")

    return FileResponse(
        path=filepath,
        filename=job["filename"],
        media_type="application/octet-stream",
    )


# ─── Background Download Worker ──────────────────────────────────────────────

def _run_download(job_id: str, url: str, format_id: str):
    """
    Runs in a background thread.
    Uses yt-dlp to download the video and pushes progress events to the job queue.
    """
    job_queue: queue.Queue = jobs[job_id]["queue"]

    def progress_hook(d):
        """Called by yt-dlp on each download chunk."""
        status = d.get("status")

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            percent = round((downloaded / total * 100), 1) if total > 0 else 0
            speed = d.get("speed") or 0
            eta = d.get("eta") or 0

            job_queue.put({
                "type": "progress",
                "percent": percent,
                "downloaded": downloaded,
                "total": total,
                "speed": round(speed / (1024 * 1024), 2) if speed else 0,  # MB/s
                "eta": eta,
            })

        elif status == "finished":
            job_queue.put({"type": "processing"})

        elif status == "error":
            job_queue.put({
                "type": "error",
                "message": str(d.get("error", "Unknown download error")),
            })

    # yt-dlp format selection:
    # "best[ext=mp4]"  → single-file MP4, no ffmpeg needed
    # "bestvideo+bestaudio" → highest quality but requires ffmpeg to merge
    if format_id and format_id != "best":
        fmt = format_id
    else:
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    ydl_opts = {
    "format": fmt,
    "outtmpl": str(DOWNLOADS_DIR / f"{job_id}_%(title)s.%(ext)s"),
    "progress_hooks": [progress_hook],
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "merge_output_format": "mp4",
    "windowsfilenames": True,
    "extractor_args": {
        "youtube": {
            "player_client": ["ios", "android"],
        }
    },
}


    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Locate the output file — glob is more reliable than predict
            pattern = str(DOWNLOADS_DIR / f"{job_id}_*")
            matched_files = glob.glob(pattern)

            if not matched_files:
                raise FileNotFoundError("Downloaded file not found in downloads folder.")

            # Pick the largest file (in case of partial segments)
            filepath = max(matched_files, key=os.path.getsize)
            filename = os.path.basename(filepath)

            with jobs_lock:
                jobs[job_id]["status"] = "finished"
                jobs[job_id]["filepath"] = filepath
                jobs[job_id]["filename"] = filename

            job_queue.put({
                "type": "finished",
                "filename": filename,
                "title": info.get("title", filename),
                "filesize": os.path.getsize(filepath),
            })

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).replace("ERROR: ", "")
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = error_msg
        job_queue.put({"type": "error", "message": error_msg})

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        job_queue.put({"type": "error", "message": str(e)})


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print()
    print("╔══════════════════════════════════════╗")
    print("║      VideoExtracterFromURL           ║")
    print("║      http://localhost:8000           ║")
    print("╚══════════════════════════════════════╝")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")