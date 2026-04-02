# export_onnx.py — ONNX export + optional quantization + benchmark

import os
import time
import json
import numpy as np
import tensorflow as tf
import onnx
import onnxruntime as ort

from src.utils import get_logger, load_config
from data_loader import get_data_generators

logger = get_logger("export_onnx")


# ---------------------------------------------------------------------------
# Optional ONNX Runtime quantization import
# ---------------------------------------------------------------------------

QUANTIZATION_AVAILABLE = True
QUANT_IMPORT_ERROR = None

try:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        CalibrationDataReader,
        QuantFormat,
    )
except Exception as e:
    QUANTIZATION_AVAILABLE = False
    QUANT_IMPORT_ERROR = str(e)

    class CalibrationDataReader:
        """Fallback placeholder when quantization imports are unavailable."""
        pass


# ---------------------------------------------------------------------------
# TF → ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(model, onnx_path: str, image_size: tuple = (150, 150)):
    import tf2onnx
    import tf2onnx.convert

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    input_signature = [
        tf.TensorSpec(
            shape=(None, *image_size, 3),
            dtype=tf.float32,
            name="input",
        )
    ]

    logger.info(f"Exporting model to ONNX → {onnx_path}")
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_path,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"ONNX export successful — size: {size_mb:.2f} MB")
    return onnx_path


# ---------------------------------------------------------------------------
# Dynamic Quantization (optional)
# ---------------------------------------------------------------------------

def dynamic_quantize(onnx_path: str, output_path: str):
    if not QUANTIZATION_AVAILABLE:
        logger.warning(f"Dynamic quantization skipped: {QUANT_IMPORT_ERROR}")
        return None

    logger.info(f"Applying Dynamic Quantization → {output_path}")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Dynamic quantized model — size: {size_mb:.2f} MB")
    return output_path


# ---------------------------------------------------------------------------
# Static Quantization (optional)
# ---------------------------------------------------------------------------

class MRICalibrationReader(CalibrationDataReader):
    """Feeds calibration batches to the static quantizer."""

    def __init__(self, data_generator, n_batches: int = 10):
        self.data = []
        self.index = 0

        for i, (batch_x, _) in enumerate(data_generator):
            if i >= n_batches:
                break
            self.data.append(batch_x.astype(np.float32))

        logger.info(f"Calibration reader: {len(self.data)} batches loaded")

    def get_next(self):
        if self.index >= len(self.data):
            return None

        batch = {"input": self.data[self.index]}
        self.index += 1
        return batch


def static_quantize(onnx_path: str, output_path: str, train_data, n_batches: int = 10):
    if not QUANTIZATION_AVAILABLE:
        logger.warning(f"Static quantization skipped: {QUANT_IMPORT_ERROR}")
        return None

    logger.info(f"Applying Static Quantization → {output_path}")
    reader = MRICalibrationReader(train_data, n_batches=n_batches)

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Static quantized model — size: {size_mb:.2f} MB")
    return output_path


# ---------------------------------------------------------------------------
# ONNX Runtime inference helper
# ---------------------------------------------------------------------------

def onnx_predict(onnx_path: str, img_array: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess.run([output_name], {input_name: img_array.astype(np.float32)})[0]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_models(tf_model, paths: dict, test_data, n_samples: int = 200) -> dict:
    logger.info("\nBenchmarking model formats ...")

    X_all, y_all = [], []
    total = 0

    for batch_x, batch_y in test_data:
        X_all.append(batch_x)
        y_all.append(batch_y)
        total += len(batch_x)
        if total >= n_samples:
            break

    X = np.concatenate(X_all, axis=0)[:n_samples].astype(np.float32)
    y = np.concatenate(y_all, axis=0)[:n_samples]
    y_true = np.argmax(y, axis=1)

    results = {}

    # TensorFlow model
    t0 = time.time()
    preds = tf_model.predict(X, verbose=0)
    tf_ms = (time.time() - t0) * 1000 / len(X)
    tf_acc = (np.argmax(preds, axis=1) == y_true).mean()

    results["TensorFlow (FP32)"] = {
        "latency_ms": float(tf_ms),
        "accuracy": float(tf_acc),
        "size_mb": None,
    }
    logger.info(f"TensorFlow (FP32) | acc={tf_acc:.4f} | {tf_ms:.2f} ms/sample")

    # ONNX models
    for name, path in paths.items():
        if path is None:
            logger.warning(f"Skipping {name} — path is None")
            continue
        if not os.path.exists(path):
            logger.warning(f"Skipping {name} — file not found: {path}")
            continue

        try:
            t0 = time.time()
            preds = onnx_predict(path, X)
            ms = (time.time() - t0) * 1000 / len(X)
            acc = (np.argmax(preds, axis=1) == y_true).mean()
            size = os.path.getsize(path) / (1024 * 1024)

            results[name] = {
                "latency_ms": float(ms),
                "accuracy": float(acc),
                "size_mb": float(size),
            }
            logger.info(f"{name:<24} | acc={acc:.4f} | {ms:.2f} ms/sample | {size:.2f} MB")

        except Exception as e:
            logger.warning(f"Skipping {name} due to runtime error: {e}")

    return results


def print_benchmark_table(results: dict):
    print("\n" + "=" * 70)
    print(f"{'Format':<26} {'Accuracy':>10} {'Latency(ms)':>13} {'Size(MB)':>12}")
    print("=" * 70)
    for name, r in results.items():
        size = f"{r['size_mb']:.2f}" if r["size_mb"] is not None else "—"
        print(f"{name:<26} {r['accuracy']:>10.4f} {r['latency_ms']:>13.2f} {size:>12}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    image_size = tuple(cfg["data"]["image_size"])
    onnx_dir = cfg["models"]["onnx_dir"]
    save_dir = cfg["models"]["save_dir"]

    os.makedirs(onnx_dir, exist_ok=True)

    train_data, val_data, test_data = get_data_generators(cfg)

    # Load best saved model
    model_path = os.path.join(save_dir, "ft_best.h5")
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Output paths
    onnx_fp32_path = os.path.join(onnx_dir, "model_fp32.onnx")
    onnx_dynamic_path = os.path.join(onnx_dir, "model_dynamic_int8.onnx")
    onnx_static_path = os.path.join(onnx_dir, "model_static_int8.onnx")

    # Export FP32 ONNX
    export_to_onnx(model, onnx_fp32_path, image_size)

    # Quantization
    dynamic_path = dynamic_quantize(onnx_fp32_path, onnx_dynamic_path)
    static_path = static_quantize(onnx_fp32_path, onnx_static_path, train_data, n_batches=50)

    # Benchmark available formats
    paths = {
        "ONNX FP32": onnx_fp32_path,
        "ONNX Dynamic INT8": dynamic_path,
        "ONNX Static INT8": static_path,
    }

    results = benchmark_models(model, paths, test_data)
    print_benchmark_table(results)

    # Save benchmark results
    bench_path = os.path.join(onnx_dir, "benchmark_results.json")
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Benchmark results saved → {bench_path}")

    if QUANTIZATION_AVAILABLE:
        logger.info("ONNX export complete. Quantization attempted.")
    else:
        logger.warning(f"ONNX export complete. Quantization skipped: {QUANT_IMPORT_ERROR}")