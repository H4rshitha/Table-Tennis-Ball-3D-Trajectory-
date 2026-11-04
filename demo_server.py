import os
import json
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import paths
from interface import TableTennisPipeline
from run_demo_pipeline import run_pipeline_analysis

app = FastAPI(title="3D Table Tennis Trajectory & Spin Uplifting Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, 'results')
EXAMPLE_IMAGES_DIR = os.path.join(WORKSPACE_ROOT, 'example_images')
STATIC_DIR = os.path.join(WORKSPACE_ROOT, 'web')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static web directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results_files", StaticFiles(directory=RESULTS_DIR), name="results_files")
app.mount("/example_images_files", StaticFiles(directory=EXAMPLE_IMAGES_DIR), name="example_images_files")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Web interface initializing... Please refresh in a moment.</h1>"

@app.get("/api/results")
async def get_results():
    json_path = os.path.join(RESULTS_DIR, 'pipeline_results.json')
    if not os.path.exists(json_path):
        # Run pipeline if not already run
        run_pipeline_analysis(EXAMPLE_IMAGES_DIR, RESULTS_DIR, fps=60.0)
    with open(json_path, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/frames")
async def get_frames_list():
    files = sorted([f for f in os.listdir(EXAMPLE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    urls = [f"/example_images_files/{f}" for f in files]
    return {"total_frames": len(files), "frame_urls": urls}

@app.post("/api/run_demo")
async def run_demo():
    try:
        results = run_pipeline_analysis(EXAMPLE_IMAGES_DIR, RESULTS_DIR, fps=60.0)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    print(f"Starting Uplifting Table Tennis Interactive Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
