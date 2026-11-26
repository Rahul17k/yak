from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, UnidentifiedImageError

from predict_yakshagana import DEFAULT_ARTIFACT_DIR, VeshaPredictor


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Yakshagana dataset"
METRICS_PATH = DEFAULT_ARTIFACT_DIR / "evaluation.json"
NOVELTY_EMBEDDINGS_PATH = DEFAULT_ARTIFACT_DIR / "novelty_training_embeddings.npy"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not NOVELTY_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Novelty reference embeddings not found: {NOVELTY_EMBEDDINGS_PATH}")
    app.state.predictor = VeshaPredictor()
    app.state.metrics = metrics
    app.state.classes = [item["label"] for item in metrics["classes"]]
    app.state.references = [metrics["references"][label] for label in app.state.classes]
    app.state.novelty_embeddings = np.load(NOVELTY_EMBEDDINGS_PATH, allow_pickle=False)
    app.state.novelty_threshold = float(metrics["novelty_screen"]["threshold"])
    yield


app = FastAPI(title="Yakshagana Vesha Identifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(request: Request):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from error

    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_FORMATS.values():
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    chunks = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
        chunks.append(chunk)
    image_bytes = b"".join(chunks)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        with Image.open(BytesIO(image_bytes)) as check:
            image_format = check.format
            check.verify()
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="The file content is not a supported JPG, PNG, or WEBP image.",
            )
        if ALLOWED_FORMATS[image_format] != content_type:
            raise HTTPException(
                status_code=415,
                detail="The image content does not match its declared file type.",
            )
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if min(image.size) < 32:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too small. Use an image at least 32 pixels wide and tall.",
                )
            result = request.app.state.predictor.predict_with_embedding(image)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        ) from error

    metrics = request.app.state.metrics
    other_class = request.app.state.predictor.other_class
    candidate = result["predicted_class"]
    confidence = result["confidence"]
    embedding = result.pop("embedding")
    dataset_similarity = float(np.max(request.app.state.novelty_embeddings @ embedding))
    score_by_class = {score["class"]: score["score"] for score in result["class_scores"]}

    if dataset_similarity < request.app.state.novelty_threshold:
        status = "not_yakshagana"
        predicted_class = None
        candidate_class = None
        confidence = None
    elif other_class and score_by_class[other_class] >= 0.5:
        status = "not_yakshagana"
        predicted_class = None
        candidate_class = None
        confidence = None
    else:
        top_two = result["class_scores"][:2]
        margin = top_two[0]["score"] - top_two[1]["score"]
        thresholds = metrics["uncertainty_thresholds"]
        if confidence < thresholds["confidence"] or margin < thresholds["margin"]:
            status = "uncertain"
            predicted_class = None
            candidate_class = candidate
        else:
            status = "yakshagana"
            predicted_class = candidate
            candidate_class = None

    return {
        "status": status,
        "predicted_class": predicted_class,
        "candidate_class": candidate_class,
        "confidence": confidence,
        "dataset_similarity": dataset_similarity,
        "class_scores": result["class_scores"],
        "uncertainty_thresholds": metrics["uncertainty_thresholds"],
        "rejection_screen_available": True,
        "rejection_method": "positive_only_novelty_screen",
        "confidence_notice": metrics["confidence_notice"],
    }


@app.get("/api/metrics")
async def get_metrics(request: Request):
    return request.app.state.metrics


@app.get("/api/references")
async def get_references(request: Request):
    return [
        {"label": label, "url": f"/api/references/{index}"}
        for index, label in enumerate(request.app.state.classes)
    ]


@app.get("/api/references/{index}")
async def get_reference(index: int, request: Request):
    references = request.app.state.references
    if index < 0 or index >= len(references):
        raise HTTPException(status_code=404, detail="Reference image not found.")

    path = (DATASET_DIR / references[index]).resolve()
    if not path.is_relative_to(DATASET_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Reference image not found.")
    return FileResponse(path)


frontend_dist = ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
