# Brain Tumor Classification

MRI brain tumor classification pipeline with TF, ONNX, Quantization, MLflow, FastAPI, and Grad-CAM XAI.

## Project Structure

```
brain_tumor_project/
├── config.yaml          ← all hyperparameters & paths
├── data_loader.py       ← dataset + augmentation
├── models.py            ← all architectures
├── train.py             ← training loop + MLflow
├── evaluate.py          ← confusion matrix + metrics
├── export_onnx.py       ← ONNX export + quantization
├── predict.py           ← inference engine (TF + ONNX)
├── save_model.py        ← save best model + Grad-CAM
├── utils.py             ← shared helpers + Grad-CAM
├── main.py              ← FastAPI backend
├── index.html           ← HTML/CSS/JS frontend
├── requirements.txt
└── data/
    ├── Training/
    └── Testing/
```

## Setup

```bash
conda activate brain_tumor
pip install -r requirements.txt
```

## Run order

```bash
# 1. Train all models
python train.py

# 2. Evaluate best model
python evaluate.py

# 3. Export to ONNX + quantize
python export_onnx.py

# 4. Start API
python main.py

# 5. Open index.html in browser
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /health         | Health check |
| GET  | /models/info    | Model metadata |
| GET  | /benchmark      | ONNX benchmark results |
| POST | /predict        | TF FP32 prediction |
| POST | /predict/onnx   | ONNX FP32 prediction |
| POST | /predict/dynamic| Dynamic INT8 prediction |
| POST | /predict/static | Static INT8 prediction |
| POST | /predict/gradcam| TF prediction + Grad-CAM |

## Models

- Baseline CNN (3-layer)
- MobileNetV2 Transfer Learning
- MobileNetV2 Fine-Tuned
- EfficientNetB0
- Optuna-tuned CNN

## Quantization

| Format | Size | Latency | Notes |
|--------|------|---------|-------|
| TF FP32 | ~20MB | baseline | Full precision |
| ONNX FP32 | ~20MB | ~1.2× faster | Optimised graph |
| Dynamic INT8 | ~5MB | ~1.5× faster | Weights only |
| Static INT8 | ~5MB | ~2-3× faster | Best for prod |
