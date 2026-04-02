# predict.py — inference engine: TF | ONNX FP32 | Dynamic INT8 | Static INT8

import os
import numpy as np
import cv2
import tensorflow as tf
import onnxruntime as ort
from tensorflow.keras.preprocessing.image import load_img

from src.utils import get_logger, get_gradcam_heatmap, get_last_conv_layer

logger = get_logger("predict")


class BrainTumorPredictor:
    """
    Unified predictor supporting TF model, ONNX FP32,
    ONNX Dynamic INT8, and ONNX Static INT8.
    """

    BACKENDS = ["tensorflow", "onnx_fp32", "onnx_dynamic", "onnx_static"]

    def __init__(self, cfg: dict, backend: str = "tensorflow"):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")

        self.backend = backend
        self.image_size = tuple(cfg["data"]["image_size"])
        self.class_names = cfg["data"]["classes"]
        self.save_dir = cfg["models"]["save_dir"]
        self.onnx_dir = cfg["models"]["onnx_dir"]

        self.tf_model = None
        self.ort_session = None
        self._load(backend)

    def _load(self, backend: str):
        if backend == "tensorflow":
            path = os.path.join(self.save_dir, "ft_best.h5")
            logger.info(f"Loading TF model from {path}")
            self.tf_model = tf.keras.models.load_model(path, compile=False)

        elif backend == "onnx_fp32":
            path = os.path.join(self.onnx_dir, "model_fp32.onnx")
            logger.info(f"Loading ONNX FP32 from {path}")
            self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        elif backend == "onnx_dynamic":
            path = os.path.join(self.onnx_dir, "model_dynamic_int8.onnx")
            logger.info(f"Loading ONNX Dynamic INT8 from {path}")
            try:
                self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise RuntimeError(
                    f"ONNX Dynamic INT8 model is not supported in this ONNX Runtime build: {e}"
                )

        elif backend == "onnx_static":
            path = os.path.join(self.onnx_dir, "model_static_int8.onnx")
            logger.info(f"Loading ONNX Static INT8 from {path}")
            try:
                self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise RuntimeError(
                    f"ONNX Static INT8 model is not supported in this ONNX Runtime build: {e}"
                )

    def preprocess(self, image_path: str) -> tuple:
        img = load_img(image_path, target_size=self.image_size)
        arr = np.array(img) / 255.0
        img_input = np.expand_dims(arr, axis=0).astype(np.float32)
        return img, arr, img_input

    def predict(self, image_path: str) -> dict:
        _, _, img_input = self.preprocess(image_path)

        if self.backend == "tensorflow":
            probs = self.tf_model.predict(img_input, verbose=0)[0]
        else:
            inp_name = self.ort_session.get_inputs()[0].name
            out_name = self.ort_session.get_outputs()[0].name
            probs = self.ort_session.run([out_name], {inp_name: img_input})[0][0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        all_probs = {cls: float(p) * 100 for cls, p in zip(self.class_names, probs)}

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probs,
            "backend": self.backend,
        }

    def predict_with_gradcam(self, image_path: str) -> dict:
        if self.backend != "tensorflow":
            raise RuntimeError("Grad-CAM is only supported with tensorflow backend.")

        result = self.predict(image_path)
        _, arr, img_input = self.preprocess(image_path)

        last_conv = get_last_conv_layer(self.tf_model)
        heatmap, _ = get_gradcam_heatmap(self.tf_model, img_input, last_conv)

        heatmap_resized = cv2.resize(heatmap, self.image_size)
        heatmap_colored = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )
        overlay = cv2.addWeighted(np.uint8(255 * arr), 0.6, heatmap_colored, 0.4, 0)

        result["gradcam_overlay"] = overlay
        result["heatmap"] = heatmap_resized
        return result


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Brain Tumor MRI Predictor")
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", default="tensorflow", choices=BrainTumorPredictor.BACKENDS)
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    predictor = BrainTumorPredictor(cfg, backend=args.backend)

    if args.gradcam and args.backend == "tensorflow":
        result = predictor.predict_with_gradcam(args.image)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        img = load_img(args.image, target_size=tuple(cfg["data"]["image_size"]))
        axes[0].imshow(img)
        axes[0].set_title("Input MRI")
        axes[0].axis("off")

        axes[1].imshow(result["heatmap"], cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(result["gradcam_overlay"])
        axes[2].set_title(f"Pred: {result['predicted_class']} ({result['confidence']:.1f}%)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        result = predictor.predict(args.image)

    print("\n" + "=" * 42)
    print(f"  PREDICTION : {result['predicted_class'].upper()}")
    print(f"  CONFIDENCE : {result['confidence']:.2f}%")
    print(f"  BACKEND    : {result['backend']}")
    print("=" * 42)
    print("  All probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob / 4)
        marker = " ← predicted" if cls == result["predicted_class"] else ""
        print(f"  {cls:<15} {prob:5.1f}%  {bar}{marker}")