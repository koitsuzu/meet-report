"""
Meeting Agent Web Server
========================
FastAPI 後端，提供：
- /api/upload  — 上傳 MP4 並啟動 Multi-Agent 處理
- /api/status/{job_id} — 查詢處理進度
- /api/download/{job_id}/{filename} — 下載產出檔案
- / — 前端介面
"""

import os
import sys
import json
import uuid
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# ── Import the agent ──────────────────────────────────────────────────────────
from meeting_agent import run_meeting_agent

app = FastAPI(title="Meeting Agent", version="1.0")

# ── Job storage (in-memory for simplicity) ────────────────────────────────────
jobs: dict = {}

# ── Static files ──────────────────────────────────────────────────────────────
Path("static").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)


# ============================================================================
# Email Service
# ============================================================================

def send_email_with_attachments(
    to_email: str,
    subject: str,
    body: str,
    attachment_paths: list[str],
):
    """Send email with meeting document attachments via Gmail SMTP."""
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_email or not smtp_password:
        print("⚠️  SMTP 未設定，跳過 Email 傳送")
        return False

    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain", "utf-8"))

    for fpath in attachment_paths:
        p = Path(fpath)
        if not p.exists():
            continue
        part = MIMEBase("application", "octet-stream")
        part.set_payload(p.read_bytes())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={p.name}",
        )
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
        print(f"✅ Email 已寄送至 {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email 寄送失敗：{e}")
        return False


# ============================================================================
# Background processing task
# ============================================================================

def process_meeting_job(job_id: str, video_path: str, description: str, email: str):
    """Background task: runs the full multi-agent pipeline."""
    jobs[job_id]["status"] = "transcribing"
    jobs[job_id]["progress"] = 20

    try:
        result = run_meeting_agent(video_path, description)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = {
            "final_answer": result.get("final_answer", ""),
            "route_history": result.get("route_history", []),
            "output_files": result.get("output_files", []),
        }

        # Send email if configured
        if email:
            output_files = result.get("output_files", [])
            topic = "Meeting"
            try:
                meeting_data = json.loads(result.get("meeting_json", "{}"))
                topic = meeting_data.get("meeting_info", {}).get("topic", "Meeting")
            except Exception:
                pass

            docx_files = [f for f in output_files if f.endswith(".docx")]
            md_files = [f for f in output_files if f.endswith(".md")]

            send_email_with_attachments(
                to_email=email,
                subject=f"📋 會議記錄 — {topic}",
                body=f"您好，\n\n附件為自動生成的會議記錄。\n\n{result.get('final_answer', '')}\n\n—— Meeting Agent",
                attachment_paths=docx_files + md_files,
            )
            jobs[job_id]["email_sent"] = True

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"❌ Job {job_id} 失敗：{e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: str = Form(""),
    email: str = Form(""),
):
    """Upload a video and start processing."""
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    upload_dir = Path("uploads") / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / file.filename

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize job
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "description": description,
        "email": email,
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
        "email_sent": False,
    }

    # Start background processing
    background_tasks.add_task(
        process_meeting_job, job_id, str(video_path), description, email
    )

    return {"job_id": job_id, "message": "處理已開始"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Check job status."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated file."""
    job = jobs.get(job_id)
    if not job or not job.get("result"):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    for fpath in job["result"].get("output_files", []):
        if Path(fpath).name == filename:
            return FileResponse(fpath, filename=filename)

    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Meeting Agent</h1><p>static/index.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
