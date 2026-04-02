# save_model.py — save best model, metadata, and Grad-CAM artifacts

import os
import json
import numpy as np
import tensorflow as tf
import mlflow

from src.utils import get_logger, load_config, generate_gradcam_overlay, get_last_conv_layer
from src.data_loader import get_data_generators

logger = get_logger("save_model")


def save_best_model(model, model_name: str, results: dict,
                    train_data, cfg: dict):
    save_dir    = cfg["models"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    image_size  = tuple(cfg["data"]["image_size"])
    class_names = cfg["data"]["classes"]

    # H5 format
    h5_path = os.path.join(save_dir, "best_brain_tumor_model.h5")
    model.save(h5_path)
    logger.info(f"Model saved (H5) → {h5_path}")

    # SavedModel format
    sm_path = os.path.join(save_dir, "best_brain_tumor_model")
    model.save(sm_path)
    logger.info(f"Model saved (SavedModel) → {sm_path}")

    # Metadata
    metadata = {
        "best_model"   : model_name,
        "class_names"  : class_names,
        "class_indices": train_data.class_indices,
        "image_size"   : list(image_size),
        "all_results"  : {k: float(v) for k, v in results.items()},
    }
    meta_path = os.path.join(save_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved → {meta_path}")

    return h5_path, sm_path, meta_path


def log_gradcam_artifacts(model, train_dir: str, class_names: list,
                           image_size: tuple, run_id: str, logs_dir: str):
    last_conv = get_last_conv_layer(model)
    logger.info(f"Generating Grad-CAM for all {len(class_names)} classes ...")

    for class_name in class_names:
        folder     = os.path.join(train_dir, class_name)
        sample_img = os.path.join(folder, os.listdir(folder)[0])
        save_path  = os.path.join(logs_dir, f"gradcam_{class_name}.png")

        generate_gradcam_overlay(model, sample_img, last_conv,
                                 image_size, class_names, save_path=save_path)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(save_path, artifact_path=f"gradcam/{class_name}")

        logger.info(f"  {class_name} Grad-CAM logged.")
