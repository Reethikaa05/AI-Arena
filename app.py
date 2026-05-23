"""
Main FastAPI Application - AI Assistant Evaluation Platform
Serves both assistants via REST API and a beautiful web UI.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing required packages...")
    os.system("pip install fastapi uvicorn pydantic --break-system-packages -q")
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

from oss_assistant.assistant import OSSAssistant
from frontier_assistant.assistant import FrontierAssistant

app = FastAPI(title="AI Assistant Evaluation Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize assistants (one session per server instance)
oss_assistant = OSSAssistant()
frontier_assistant = FrontierAssistant()

# In-memory eval results store
eval_store = {"results": None, "summary": None, "running": False, "progress": 0}


class ChatRequest(BaseModel):
    message: str
    model: str  # "oss" or "frontier"


class ResetRequest(BaseModel):
    model: str  # "oss", "frontier", or "both"


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>AI Assistant Platform</h1><p>static/index.html not found</p>")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if req.model == "oss":
        result = oss_assistant.chat(req.message)
        result["assistant"] = "oss"
    elif req.model == "frontier":
        result = frontier_assistant.chat(req.message)
        result["assistant"] = "frontier"
    elif req.model == "both":
        oss_result = oss_assistant.chat(req.message)
        frontier_result = frontier_assistant.chat(req.message)
        return JSONResponse({
            "oss": {**oss_result, "assistant": "oss"},
            "frontier": {**frontier_result, "assistant": "frontier"},
        })
    else:
        raise HTTPException(status_code=400, detail="model must be 'oss', 'frontier', or 'both'")

    return JSONResponse(result)


@app.post("/api/reset")
async def reset(req: ResetRequest):
    if req.model in ("oss", "both"):
        oss_assistant.reset_conversation()
    if req.model in ("frontier", "both"):
        frontier_assistant.reset_conversation()
    return {"status": "ok", "reset": req.model}


@app.get("/api/metrics")
async def metrics():
    return {
        "oss": oss_assistant.get_metrics(),
        "frontier": frontier_assistant.get_metrics(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/eval/results")
async def eval_results():
    results_path = Path(__file__).parent / "evaluation" / "summary.json"
    if results_path.exists():
        with open(results_path) as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "No evaluation results found. Run evaluation first."})


@app.get("/api/eval/raw")
async def eval_raw():
    results_path = Path(__file__).parent / "evaluation" / "raw_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "No raw results found."})


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "oss_model": "Qwen2.5-72B-Instruct",
        "frontier_model": "claude-sonnet-4-20250514",
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
