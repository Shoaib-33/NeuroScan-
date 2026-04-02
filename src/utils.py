# utils.py — shared helpers used across all modules

import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history, title: str, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{title} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plot_comparison(results: dict, save_path: str = None):
    plt.figure(figsize=(10, 4))
    colors = [
        "crimson" if v == max(results.values()) else "steelblue"
        for v in results.values()
    ]
    bars = plt.bar(results.keys(), results.values(), color=colors)
    plt.bar_label(bars, fmt="%.4f", padding=3)
    plt.ylim(min(results.values()) - 0.05, 1.0)
    plt.title("Model Comparison - Validation Accuracy (red = best)")
    plt.ylabel("Val Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------------------

def _collect_all_layers(model) -> list:
    """
    Flatten all layers from a model including layers inside nested sub-models.
    Returns a flat list of (layer_object, parent_model) tuples.
    """
    result = []

    def _recurse(m):
        for layer in m.layers:
            result.append(layer)
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                _recurse(layer)

    _recurse(model)
    return result


def get_last_conv_layer(model) -> str:
    """
    Return the name of the last Conv2D layer found anywhere inside the model,
    including inside nested sub-models (MobileNetV2, EfficientNetB0 etc.).
    """
    all_layers  = _collect_all_layers(model)
    conv_layers = [l for l in all_layers if isinstance(l, tf.keras.layers.Conv2D)]

    if not conv_layers:
        raise ValueError("No Conv2D layer found in model.")

    return conv_layers[-1].name


def _build_gradcam_model(model, last_conv_layer_name: str):
    """
    Build a Grad-CAM sub-model that outputs:
      [conv_layer_output, final_model_predictions]

    Works for both:
      - Plain CNNs: Conv2D layers are direct children of the model
      - Nested models: Conv2D is inside a sub-model (MobileNetV2, EfficientNetB0)

    Strategy: find which sub-model owns the target conv layer, build a
    feature extractor from that sub-model's input to [conv_output, sub_output],
    then chain it with the remaining head layers of the outer model.
    """
    all_layers = _collect_all_layers(model)

    # Find the layer object
    target_layer = None
    for layer in all_layers:
        if layer.name == last_conv_layer_name:
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    # Check if the conv layer is a direct child of the outer model
    direct_names = [l.name for l in model.layers]

    if last_conv_layer_name in direct_names:
        # Plain CNN — simple case
        grad_model = tf.keras.models.Model(
            inputs  = model.input,
            outputs = [model.get_layer(last_conv_layer_name).output, model.output]
        )
        return grad_model, None   # None = no separate head needed

    # Nested model case — find which direct child sub-model contains the layer
    owner_submodel = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            sub_names = [l.name for l in _collect_all_layers(layer)]
            if last_conv_layer_name in sub_names:
                owner_submodel = layer
                break

    if owner_submodel is None:
        raise ValueError(
            f"Could not find parent sub-model for layer '{last_conv_layer_name}'."
        )

    # Build: sub-model input -> [conv_output, sub_model_output]
    sub_grad_model = tf.keras.models.Model(
        inputs  = owner_submodel.input,
        outputs = [
            owner_submodel.get_layer(last_conv_layer_name).output,
            owner_submodel.output,
        ]
    )

    # Collect head layers (everything after the sub-model in the outer model)
    head_layers = []
    found = False
    for layer in model.layers:
        if found:
            head_layers.append(layer)
        if layer.name == owner_submodel.name:
            found = True

    return sub_grad_model, head_layers


def get_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str):
    """
    Compute Grad-CAM heatmap.

    Parameters
    ----------
    model               : compiled Keras model
    img_array           : preprocessed image, shape (1, H, W, 3), values in [0,1]
    last_conv_layer_name: name of the target Conv2D layer

    Returns
    -------
    heatmap   : np.ndarray shape (H_conv, W_conv), values in [0,1]
    pred_idx  : int, predicted class index
    """
    grad_model, head_layers = _build_gradcam_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:

        if head_layers is None:
            # Plain CNN — single forward pass
            conv_outputs, predictions = grad_model(img_array)
        else:
            # Nested model — two-stage forward pass
            conv_outputs, sub_output = grad_model(img_array)

            # Run through head layers sequentially
            x = sub_output
            for layer in head_layers:
                x = layer(x)
            predictions = x

        pred_idx = tf.argmax(predictions[0])
        loss      = predictions[:, pred_idx]

        # Watch conv_outputs so we can compute gradients w.r.t. it
        tape.watch(conv_outputs)

    # Recompute with watched tensor inside tape scope
    with tf.GradientTape() as tape2:
        tape2.watch(conv_outputs)

        if head_layers is None:
            conv_out_val, preds = grad_model(img_array)
        else:
            conv_out_val, sub_out = grad_model(img_array)
            x = sub_out
            for layer in head_layers:
                x = layer(x)
            preds = x

        pred_idx  = int(tf.argmax(preds[0]))
        class_loss = preds[:, pred_idx]

    grads = tape2.gradient(class_loss, conv_out_val)

    if grads is None:
        raise ValueError(
            "Gradients are None. The conv layer output is not part of the "
            "computation graph. Try a different layer name."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_out_val[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), pred_idx


# ---------------------------------------------------------------------------
# Full Grad-CAM visualisation
# ---------------------------------------------------------------------------

def generate_gradcam_overlay(model, img_path: str, last_conv_layer: str,
                              image_size: tuple, class_names: list,
                              save_path: str = None):
    img       = load_img(img_path, target_size=image_size)
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)

    heatmap, pred_idx = get_gradcam_heatmap(model, img_input, last_conv_layer)

    heatmap_resized = cv2.resize(heatmap, image_size)
    heatmap_colored = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    overlay = cv2.addWeighted(
        np.uint8(255 * img_array), 0.6, heatmap_colored, 0.4, 0
    )

    probs = model.predict(img_input, verbose=0)[0]
    conf  = probs[pred_idx] * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred: {class_names[pred_idx]} ({conf:.1f}%)")
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM - {class_names[pred_idx].upper()}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()

    return pred_idx, conf, overlay