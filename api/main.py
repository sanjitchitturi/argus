"""
Argus FastAPI Backend
Endpoints: /inspect, /detect, /caption, /vqa, /report, /evaluate, /health
"""

from __future__ import annotations
import io
import uuid
from threading import Lock
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from api.request_security import RequestSecurityMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

# Lazy-init pipeline (loaded on first request or startup)
_pipeline = None
_pipeline_lock = Lock()


def get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from src.pipeline import ArgusPipeline
            from src.utils.config import load_config

            cfg = load_config()
            _pipeline = ArgusPipeline(
                yolo_model=cfg["detection"]["model"],
                vlm_model=cfg["vlm"]["model"],
                llm_model=cfg["llm"]["model"],
                device=cfg["detection"]["device"],
                detection_confidence=cfg["detection"]["confidence"],
            )
    return _pipeline


# ── App Setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Argus Inspection API",
    description="Vision-Language Industrial Defect Inspection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    RequestSecurityMiddleware,
    allowed_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    max_bytes=11 * 1024 * 1024,
)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cls: str = Field(alias="class")
    confidence: float
    bbox: BBox
    area_px: int
    severity: str


class DetectionResponse(BaseModel):
    image_id: str
    detections: List[dict]
    summary: str
    num_defects: int
    inference_ms: float
    image_size: dict


class CaptionResponse(BaseModel):
    image_id: str
    caption: str
    inference_ms: float
    model: str


class VQARequest(BaseModel):
    question: str
    image_b64: str


class VQAResponse(BaseModel):
    image_id: str
    question: str
    answer: str
    inference_ms: float


class InspectionResponse(BaseModel):
    image_id: str
    timestamp: str
    detection: dict
    caption: dict
    vqa: List[dict]
    report: dict
    pipeline_metadata: dict


class EvalRequest(BaseModel):
    predictions: List[dict] = Field(max_length=1000)
    ground_truths: List[dict] = Field(max_length=1000)
    captions: List[str] = []
    reference_captions: List[List[str]] = []
    reports: List[str] = []
    reference_reports: List[str] = []


# ── Helpers ────────────────────────────────────────────────────────────────


def decode_image(file: UploadFile) -> Image.Image:
    try:
        content = file.file.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "Image exceeds 10 MB")
        opened = Image.open(io.BytesIO(content))
        if opened.width * opened.height > 20_000_000:
            raise HTTPException(413, "Image exceeds 20 megapixels")
        return opened.convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image") from e


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    import os

    return {
        "mode": os.getenv("ARGUS_MODE", "demo"),
        "synthetic": os.getenv("ARGUS_MODE", "demo") == "demo",
        "status": "healthy",
        "service": "Argus Inspection API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
def detect_defects(
    file: UploadFile = File(...),
    image_id: Optional[str] = Form(None),
):
    """Run YOLOv8 defect detection on an uploaded image."""
    img = decode_image(file)
    img_id = image_id or f"ARGUS-{uuid.uuid4().hex[:8].upper()}"
    from src.utils.config import validate_image_id

    try:
        validate_image_id(img_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    pipeline = get_pipeline()
    result = pipeline.detector.detect(img)
    return {**result, "image_id": img_id}


@app.post("/caption", response_model=CaptionResponse, tags=["VLM"])
def generate_caption(
    file: UploadFile = File(...),
    image_id: Optional[str] = Form(None),
):
    """Generate image caption using BLIP-2 / LLaVA."""
    img = decode_image(file)
    img_id = image_id or f"ARGUS-{uuid.uuid4().hex[:8].upper()}"
    from src.utils.config import validate_image_id

    try:
        validate_image_id(img_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    pipeline = get_pipeline()
    result = pipeline.vlm.caption(img)
    return {**result, "image_id": img_id}


@app.post("/vqa", response_model=VQAResponse, tags=["VLM"])
def visual_qa(
    file: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=500),
    image_id: Optional[str] = Form(None),
):
    """Answer a question about the uploaded image."""
    img = decode_image(file)
    img_id = image_id or f"ARGUS-{uuid.uuid4().hex[:8].upper()}"
    from src.utils.config import validate_image_id

    try:
        validate_image_id(img_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    pipeline = get_pipeline()
    result = pipeline.vlm.answer(img, question)
    return {**result, "image_id": img_id}


@app.post("/inspect", tags=["Pipeline"])
def full_inspection(
    file: UploadFile = File(...),
    image_id: Optional[str] = Form(None),
    inspector: str = Form("Reviewer", max_length=100),
    questions: Optional[str] = Form(None, max_length=6000),  # JSON-encoded list
    save_results: bool = Form(True),
):
    """
    Run the full Argus pipeline:
    Detection → Caption → VQA → Report Generation
    """
    img = decode_image(file)
    img_id = image_id or f"ARGUS-{uuid.uuid4().hex[:8].upper()}"
    from src.utils.config import validate_image_id

    try:
        validate_image_id(img_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    custom_q = None
    if questions:
        try:
            custom_q = json.loads(questions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="questions must be valid JSON array")

    if custom_q is not None and (
        not isinstance(custom_q, list)
        or len(custom_q) > 10
        or any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in custom_q)
    ):
        raise HTTPException(
            422,
            "questions must be a list of at most 10 nonempty strings, up to 500 characters each",
        )
    pipeline = get_pipeline()
    result = pipeline.run(
        image=img,
        image_id=img_id,
        inspector=inspector,
        custom_questions=custom_q,
        save_results=save_results,
    )
    return JSONResponse(content=result)


@app.post("/visualize", tags=["Detection"])
def visualize_detections(file: UploadFile = File(...)):
    """Return annotated image with detection bounding boxes as PNG."""
    import cv2

    img = decode_image(file)
    pipeline = get_pipeline()
    annotated = pipeline.visualize(img)
    # Convert RGB → BGR → encode PNG
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.post("/evaluate", tags=["Evaluation"])
def evaluate(request: EvalRequest):
    """Compute detection, captioning, and report metrics."""
    from src.evaluation.metrics import run_full_evaluation

    try:
        metrics = run_full_evaluation(
            predictions=request.predictions,
            ground_truths=request.ground_truths,
            captions=request.captions,
            reference_captions=request.reference_captions,
            reports=request.reports,
            reference_reports=request.reference_reports,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return metrics


@app.get("/experiments", tags=["Results"])
def list_experiments():
    """List saved inspection results."""
    from src.utils.config import EXPERIMENTS_DIR

    files = sorted(
        Path(EXPERIMENTS_DIR).glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    results = []
    for f in files[:50]:  # max 50
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append(
                {
                    "image_id": data.get("image_id"),
                    "timestamp": data.get("timestamp"),
                    "num_defects": data.get("detection", {}).get("num_defects", 0),
                    "file": f.name,
                }
            )
        except Exception:
            pass
    return {"experiments": results, "count": len(results)}


@app.get("/experiments/{image_id}", tags=["Results"])
def get_experiment(image_id: str):
    """Retrieve a specific inspection result by image ID."""
    from src.utils.config import EXPERIMENTS_DIR
    from src.utils.config import validate_image_id

    try:
        validate_image_id(image_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    path = Path(EXPERIMENTS_DIR) / f"{image_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {image_id} not found")
    with open(path) as f:
        return json.load(f)


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
