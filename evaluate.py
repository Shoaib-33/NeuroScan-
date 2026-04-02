# evaluate.py — confusion matrix, classification report, per-class metrics

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from utils import get_logger, load_config
from src.data_loader import get_data_generators

logger = get_logger("evaluate")


def evaluate_model(model, test_data, class_names: list,
                   model_name: str = "model", save_dir: str = "./logs"):
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Evaluating {model_name} on test set ...")

    # Reset generator so it always starts from the beginning
    test_data.reset()

    y_pred_prob = model.predict(test_data, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_data.classes

    # Align lengths — generator may yield slightly more due to batch rounding
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[:min_len]
    y_pred  = y_pred[:min_len]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logger.info(f"\nClassification Report - {model_name}:\n{report}")

    report_path = os.path.join(save_dir, f"{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    logger.info(f"Report saved -> {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info(f"Confusion matrix saved -> {cm_path}")

    # Use sklearn accuracy (more reliable than model.evaluate on generators)
    test_acc = float((y_pred == y_true).mean())
    logger.info(f"{model_name} - Test accuracy: {test_acc:.4f}")

    return {
        "test_accuracy" : test_acc,
        "y_true"        : y_true.tolist(),
        "y_pred"        : y_pred.tolist(),
    }


def evaluate_all_models(models_dict: dict, test_data, class_names: list,
                        save_dir: str = "./logs"):
    results = {}
    for name, model in models_dict.items():
        safe_name     = name.lower().replace(" ", "_")
        results[name] = evaluate_model(
            model, test_data, class_names,
            model_name=safe_name, save_dir=save_dir,
        )

    logger.info("\n" + "=" * 45)
    logger.info(f"{'Model':<25} {'Test Acc':>10}")
    logger.info("=" * 45)
    for name, r in results.items():
        logger.info(f"{name:<25} {r['test_accuracy']:>10.4f}")
    logger.info("=" * 45)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    _, _, test_data = get_data_generators(cfg)

    class_names = cfg["data"]["classes"]
    save_dir    = cfg["models"]["save_dir"]
    logs_dir    = cfg["models"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)

    # Load best saved model
    best_model_path = os.path.join(save_dir, "ft_best.h5")
    if not os.path.exists(best_model_path):
        logger.error(f"Model not found at {best_model_path}. Run train.py first.")
        exit(1)

    logger.info(f"Loading model from {best_model_path}")

    # compile=False avoids optimizer state errors when loading
    # MobileNetV2 or EfficientNetB0 models in TF 2.10
    model = tf.keras.models.load_model(best_model_path, compile=False)
    model.compile(
        optimizer = "adam",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    results = evaluate_model(
        model, test_data, class_names,
        model_name="best_model", save_dir=logs_dir,
    )
    logger.info("Evaluation complete.")