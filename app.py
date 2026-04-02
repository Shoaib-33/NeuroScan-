# main.py — FastAPI backend
# Endpoints:
#   GET  /                  -> serve static/index.html
#   GET  /health            -> health check
#   GET  /benchmark         -> cached benchmark results
#   GET  /models/info       -> metadata about loaded models
#   POST /predict           -> TF model prediction
#   POST /predict/onnx      -> ONNX FP32 prediction
#   POST /predict/dynamic   -> Dynamic INT8 prediction
#   POST /predict/static    -> Static INT8 prediction
#   POST /predict/gradcam   -> TF prediction + Grad-CAM overlay image

import os
import io
import json
import time
import base64
import tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils import load_config, get_logger
from predict import BrainTumorPredictor

logger = get_logger("api")
cfg = load_config("config.yaml")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Brain Tumor Classification API",
    description = "MRI brain tumor classification with TF, ONNX FP32, Dynamic INT8, Static INT8 + Grad-CAM",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")

# ---------------------------------------------------------------------------
# Load predictors at startup
# ---------------------------------------------------------------------------

predictors:  dict = {}
load_errors: dict = {}   # stores error messages for failed backends

@app.on_event("startup")
async def load_models():
    global predictors, load_errors
    logger.info("Loading models ...")

    # TF backend — always try first
    try:
        predictors["tensorflow"] = BrainTumorPredictor(cfg, backend="tensorflow")
        logger.info("[OK] TF model loaded.")
    except Exception as e:
        load_errors["tensorflow"] = str(e)
        logger.error(f"[FAIL] TF model failed to load: {e}")

    # ONNX backends — optional, warn if not found
    for backend in ["onnx_fp32", "onnx_dynamic", "onnx_static"]:
        try:
            predictors[backend] = BrainTumorPredictor(cfg, backend=backend)
            logger.info(f"[OK] {backend} loaded.")
        except FileNotFoundError as e:
            load_errors[backend] = str(e)
            logger.warning(f"[SKIP] {backend} — file not found. Run export_onnx.py first.")
        except Exception as e:
            load_errors[backend] = str(e)
            logger.error(f"[FAIL] {backend} failed: {e}")

    logger.info(f"Loaded backends : {list(predictors.keys())}")
    if load_errors:
        logger.warning(f"Failed backends : {list(load_errors.keys())}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_upload_temp(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.flush()
    return tmp.name


def numpy_to_b64(img_array: np.ndarray) -> str:
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buf     = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_with_backend(backend: str, tmp_path: str) -> dict:
    if backend not in predictors:
        # Give a specific error message explaining what to do
        if backend in load_errors:
            detail = (
                f"Backend '{backend}' failed to load: {load_errors[backend]}. "
                f"Run export_onnx.py first to generate ONNX models."
            )
        else:
            detail = (
                f"Backend '{backend}' is not available. "
                f"Run export_onnx.py to generate ONNX models."
            )
        raise HTTPException(status_code=503, detail=detail)

    t0     = time.time()
    result = predictors[backend].predict(tmp_path)
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_class   : str
    confidence        : float
    all_probabilities : dict
    backend           : str
    latency_ms        : float


class GradCamResponse(BaseModel):
    predicted_class   : str
    confidence        : float
    all_probabilities : dict
    backend           : str
    latency_ms        : float
    gradcam_b64       : str
    heatmap_b64       : str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="static/index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    return {
        "status"          : "ok",
        "loaded_backends" : list(predictors.keys()),
        "failed_backends" : load_errors,   
    }


@app.get("/models/info")
async def models_info():
    meta_path = os.path.join(cfg["models"]["save_dir"], "model_metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="model_metadata.json not found.")
    with open(meta_path) as f:
        return json.load(f)


@app.get("/benchmark")
async def get_benchmark():
    bench_path = os.path.join(cfg["models"]["onnx_dir"], "benchmark_results.json")
    if not os.path.exists(bench_path):
        raise HTTPException(
            status_code=404,
            detail="benchmark_results.json not found. Run export_onnx.py first.",
        )
    with open(bench_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
async def predict_tf(file: UploadFile = File(...)):
    tmp = save_upload_temp(file)
    try:
        return predict_with_backend("tensorflow", tmp)
    finally:
        os.unlink(tmp)


@app.post("/predict/onnx", response_model=PredictionResponse)
async def predict_onnx_fp32(file: UploadFile = File(...)):
    tmp = save_upload_temp(file)
    try:
        return predict_with_backend("onnx_fp32", tmp)
    finally:
        os.unlink(tmp)


@app.post("/predict/dynamic", response_model=PredictionResponse)
async def predict_dynamic(file: UploadFile = File(...)):
    tmp = save_upload_temp(file)
    try:
        return predict_with_backend("onnx_dynamic", tmp)
    finally:
        os.unlink(tmp)


@app.post("/predict/static", response_model=PredictionResponse)
async def predict_static(file: UploadFile = File(...)):
    tmp = save_upload_temp(file)
    try:
        return predict_with_backend("onnx_static", tmp)
    finally:
        os.unlink(tmp)


@app.post("/predict/gradcam", response_model=GradCamResponse)
async def predict_gradcam(file: UploadFile = File(...)):
    if "tensorflow" not in predictors:
        raise HTTPException(status_code=503, detail="TF model not loaded.")

    tmp = save_upload_temp(file)
    try:
        t0      = time.time()
        result  = predictors["tensorflow"].predict_with_gradcam(tmp)
        latency = round((time.time() - t0) * 1000, 2)

        gradcam_b64 = numpy_to_b64(result["gradcam_overlay"])
        heatmap_b64 = numpy_to_b64(
            cv2.applyColorMap(
                np.uint8(255 * result["heatmap"]), cv2.COLORMAP_JET
            )[:, :, ::-1]
        )

        return {
            "predicted_class"  : result["predicted_class"],
            "confidence"       : result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "backend"          : result["backend"],
            "latency_ms"       : latency,
            "gradcam_b64"      : gradcam_b64,
            "heatmap_b64"      : heatmap_b64,
        }
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    api_cfg = cfg["api"]
    uvicorn.run(
        "main:app",
        host   = api_cfg["host"],
        port   = api_cfg["port"],
        reload = api_cfg["reload"],
    )