# train.py — training loop with MLflow + DagsHub tracking

import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import optimizers

from src.utils import get_logger, load_config, plot_history, plot_comparison
from src.data_loader import get_data_generators
from src.models import (build_baseline_cnn, build_mobilenet_tl, build_mobilenet_finetuned,
                    build_efficientnet, build_optuna_cnn, compile_model)
import optuna

logger = get_logger("train")


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(cfg: dict):
    ml = cfg["mlflow"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = ml["dagshub_username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ml["dagshub_token"]
    uri = f"https://dagshub.com/{ml['dagshub_username']}/{ml['dagshub_repo']}.mlflow"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(ml["experiment_name"])
    logger.info(f"MLflow → DagsHub: {uri}")


def log_epoch_metrics(history):
    """Log per-epoch metrics to the active MLflow run."""
    for epoch, (ta, va, tl, vl) in enumerate(zip(
        history.history["accuracy"], history.history["val_accuracy"],
        history.history["loss"],     history.history["val_loss"],
    )):
        mlflow.log_metrics({
            "train_accuracy" : float(ta),
            "val_accuracy"   : float(va),
            "train_loss"     : float(tl),
            "val_loss"       : float(vl),
        }, step=epoch)


def get_callbacks(cfg: dict, checkpoint_path: str):
    """
    Standard callbacks for all models.
    save_format='h5' is required to avoid EagerTensor JSON
    serialization crash with EfficientNet in TF 2.10.
    """
    t = cfg["training"]
    return [
        EarlyStopping(
            monitor             = "val_accuracy",
            patience            = t["early_stopping_patience"],
            restore_best_weights= True,
            verbose             = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = t["reduce_lr_factor"],
            patience = t["reduce_lr_patience"],
            min_lr   = t["min_lr"],
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath        = checkpoint_path,
            monitor         = "val_accuracy",
            save_best_only  = True,
            save_format     = "h5",          # ← fixes EagerTensor crash
            verbose         = 0,
        ),
    ]


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_baseline(cfg, train_data, val_data):
    image_size = tuple(cfg["data"]["image_size"])
    epochs     = cfg["training"]["epochs"]
    save_dir   = cfg["models"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    model = compile_model(build_baseline_cnn(image_size=image_size))

    with mlflow.start_run(run_name="Baseline_CNN"):
        mlflow.log_params({
            "model_type" : "Baseline CNN",
            "filters"    : "32-64-128",
            "optimizer"  : "adam",
            "lr"         : 0.001,
            "epochs"     : epochs,
            "batch_size" : cfg["data"]["batch_size"],
        })
        history = model.fit(
            train_data, epochs=epochs, validation_data=val_data,
            callbacks=get_callbacks(cfg, f"{save_dir}/baseline_best.h5")
        )
        log_epoch_metrics(history)
        mlflow.log_metrics({
            "best_val_accuracy" : float(max(history.history["val_accuracy"])),
            "best_val_loss"     : float(min(history.history["val_loss"])),
        })
        mlflow.keras.log_model(model, "baseline_cnn")
        logger.info("Baseline CNN training complete.")
    return model, history


def train_transfer_learning(cfg, train_data, val_data):
    image_size = tuple(cfg["data"]["image_size"])
    epochs     = cfg["training"]["epochs"]
    save_dir   = cfg["models"]["save_dir"]

    model = compile_model(build_mobilenet_tl(image_size=image_size))

    with mlflow.start_run(run_name="Transfer_Learning_MobileNetV2"):
        mlflow.log_params({
            "model_type"  : "MobileNetV2 TL",
            "base_frozen" : True,
            "optimizer"   : "adam",
            "lr"          : 0.001,
            "epochs"      : epochs,
        })
        history = model.fit(
            train_data, epochs=epochs, validation_data=val_data,
            callbacks=get_callbacks(cfg, f"{save_dir}/tl_best.h5")
        )
        log_epoch_metrics(history)
        mlflow.log_metrics({
            "best_val_accuracy" : float(max(history.history["val_accuracy"])),
            "best_val_loss"     : float(min(history.history["val_loss"])),
        })
        mlflow.keras.log_model(model, "transfer_learning")
        logger.info("Transfer Learning training complete.")
    return model, history


def train_finetuned(cfg, tl_model, train_data, val_data):
    epochs   = cfg["training"]["epochs"]
    save_dir = cfg["models"]["save_dir"]

    build_mobilenet_finetuned(tl_model.layers[0], unfreeze_last=20)
    compile_model(tl_model, lr=1e-5)

    with mlflow.start_run(run_name="Fine_Tuned_MobileNetV2"):
        mlflow.log_params({
            "model_type"      : "MobileNetV2 Fine-Tuned",
            "unfrozen_layers" : 20,
            "lr"              : 1e-5,
            "epochs"          : epochs,
        })
        history = tl_model.fit(
            train_data, epochs=epochs, validation_data=val_data,
            callbacks=get_callbacks(cfg, f"{save_dir}/ft_best.h5")
        )
        log_epoch_metrics(history)
        mlflow.log_metrics({
            "best_val_accuracy" : float(max(history.history["val_accuracy"])),
            "best_val_loss"     : float(min(history.history["val_loss"])),
        })
        mlflow.keras.log_model(tl_model, "fine_tuned")
        logger.info("Fine-Tuned training complete.")
    return tl_model, history


def train_efficientnet(cfg, train_data, val_data):
    """
    EfficientNetB0 in TF 2.10 crashes ModelCheckpoint because its internal
    rescaling layer stores weights as EagerTensors which cannot be JSON-serialized
    during checkpoint saving.

    Fix: use a custom callback that calls model.save_weights() instead of
    model.save() — weights-only saving never touches the model config JSON,
    so EagerTensors are never serialized.
    """
    image_size = tuple(cfg["data"]["image_size"])
    epochs     = cfg["training"]["epochs"]
    save_dir   = cfg["models"]["save_dir"]
    t          = cfg["training"]

    model = build_efficientnet(image_size=image_size)
    model.compile(
        optimizer = optimizers.Adam(learning_rate=0.001),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── Custom checkpoint: saves weights only (no JSON config serialization) ──
    weights_path   = os.path.join(save_dir, "effnet_best_weights.h5")
    best_val_acc   = [0.0]    # mutable container so inner class can write to it

    class WeightsCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            va = float(logs.get("val_accuracy", 0.0))
            if va > best_val_acc[0]:
                best_val_acc[0] = va
                self.model.save_weights(weights_path)
                logger.info(f"  EfficientNet weights saved (val_acc={va:.4f})")

    with mlflow.start_run(run_name="EfficientNetB0_TL"):
        mlflow.log_params({
            "model_type"  : "EfficientNetB0",
            "base_frozen" : True,
            "lr"          : 0.001,
            "epochs"      : epochs,
        })

        history = model.fit(
            train_data,
            epochs          = epochs,
            validation_data = val_data,
            callbacks       = [
                EarlyStopping(
                    monitor              = "val_accuracy",
                    patience             = t["early_stopping_patience"],
                    restore_best_weights = True,
                    verbose              = 1,
                ),
                ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = t["reduce_lr_factor"],
                    patience = t["reduce_lr_patience"],
                    min_lr   = t["min_lr"],
                    verbose  = 1,
                ),
                WeightsCheckpoint(),   # ← replaces ModelCheckpoint entirely
            ],
        )

        log_epoch_metrics(history)
        mlflow.log_metrics({
            "best_val_accuracy" : float(max(history.history["val_accuracy"])),
            "best_val_loss"     : float(min(history.history["val_loss"])),
        })

        # Restore best weights
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            logger.info("Best EfficientNet weights restored.")

        # mlflow.keras.log_model crashes for EfficientNetB0 in TF 2.10 because
        # mlflow internally calls model.save() which hits the same EagerTensor
        # JSON bug. Workaround: save weights manually and log as artifact.
        final_weights_path = os.path.join(save_dir, "effnet_final_weights.h5")
        model.save_weights(final_weights_path)
        mlflow.log_artifact(final_weights_path, artifact_path="efficientnet_weights")
        mlflow.set_tag("efficientnet_note",
                       "Logged as weights-only artifact due to TF2.10 EagerTensor bug")
        logger.info(f"EfficientNet weights logged to MLflow → {final_weights_path}")
        logger.info("EfficientNetB0 training complete.")
    return model, history


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def run_optuna(cfg, train_data, val_data):
    image_size = tuple(cfg["data"]["image_size"])
    n_trials   = cfg["optuna"]["n_trials"]

    def objective(trial):
        params = {
            "filters_1"   : trial.suggest_categorical("filters_1",   [32, 64]),
            "filters_2"   : trial.suggest_categorical("filters_2",   [64, 128]),
            "filters_3"   : trial.suggest_categorical("filters_3",   [128, 256]),
            "dense_units" : trial.suggest_categorical("dense_units", [64, 128, 256]),
            "dropout"     : trial.suggest_float("dropout", 0.2, 0.5),
            "lr"          : trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }

        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params(params)

            m = compile_model(
                build_optuna_cnn(params, image_size=image_size),
                lr=params["lr"]
            )
            h = m.fit(
                train_data,
                epochs          = 10,
                validation_data = val_data,
                callbacks       = [EarlyStopping(monitor="val_accuracy",
                                                 patience=3,
                                                 restore_best_weights=True)],
                verbose         = 0,
            )

            best_val = float(max(h.history["val_accuracy"]))
            mlflow.log_metric("best_val_accuracy", best_val)

            for ep, (ta, va, tl, vl) in enumerate(zip(
                h.history["accuracy"], h.history["val_accuracy"],
                h.history["loss"],     h.history["val_loss"],
            )):
                mlflow.log_metrics({
                    "train_accuracy" : float(ta),
                    "val_accuracy"   : float(va),
                    "train_loss"     : float(tl),
                    "val_loss"       : float(vl),
                }, step=ep)

        return best_val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    with mlflow.start_run(run_name="Optuna_Search_Parent"):
        mlflow.log_params({
            "n_trials"  : n_trials,
            "direction" : "maximize",
            "objective" : "val_accuracy",
        })
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_metric("best_val_accuracy", float(study.best_value))
        mlflow.log_metric("best_trial_number", study.best_trial.number)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    logger.info(f"Optuna best val_accuracy : {study.best_value:.4f}")
    logger.info(f"Optuna best params       : {study.best_params}")
    return study


def train_optuna_best(cfg, study, train_data, val_data):
    image_size = tuple(cfg["data"]["image_size"])
    epochs     = cfg["training"]["epochs"]
    save_dir   = cfg["models"]["save_dir"]
    p          = study.best_params

    model = compile_model(
        build_optuna_cnn(p, image_size=image_size),
        lr=p["lr"]
    )

    with mlflow.start_run(run_name="Optuna_Best_CNN_Final") as run:
        mlflow.log_params({
            **p,
            "model_type" : "Optuna Best CNN",
            "epochs"     : epochs,
        })
        history = model.fit(
            train_data, epochs=epochs, validation_data=val_data,
            callbacks=get_callbacks(cfg, f"{save_dir}/optuna_best.h5")
        )
        log_epoch_metrics(history)
        mlflow.log_metrics({
            "best_val_accuracy" : float(max(history.history["val_accuracy"])),
            "best_val_loss"     : float(min(history.history["val_loss"])),
        })
        sample    = train_data[0][0][:1]
        signature = infer_signature(sample, model.predict(sample))
        mlflow.keras.log_model(model, "optuna_best_cnn", signature=signature)
        best_run_id = run.info.run_id

    logger.info("Optuna Best CNN training complete.")
    return model, history, best_run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    setup_mlflow(cfg)

    train_data, val_data, test_data = get_data_generators(cfg)

    logger.info("=" * 50)
    logger.info("Starting full training pipeline")
    logger.info("=" * 50)

    baseline_model, history_base   = train_baseline(cfg, train_data, val_data)
    tl_model,       history_tl     = train_transfer_learning(cfg, train_data, val_data)
    ft_model,       history_ft     = train_finetuned(cfg, tl_model, train_data, val_data)
    eff_model,      history_eff    = train_efficientnet(cfg, train_data, val_data)
    study                          = run_optuna(cfg, train_data, val_data)
    optuna_model, history_optuna, best_run_id = train_optuna_best(
        cfg, study, train_data, val_data
    )

    results = {
        "Baseline CNN"      : float(max(history_base.history["val_accuracy"])),
        "Transfer Learning" : float(max(history_tl.history["val_accuracy"])),
        "Fine-Tuned"        : float(max(history_ft.history["val_accuracy"])),
        "EfficientNetB0"    : float(max(history_eff.history["val_accuracy"])),
        "Optuna Best CNN"   : float(max(history_optuna.history["val_accuracy"])),
    }

    logger.info("\n" + "=" * 45)
    logger.info(f"{'Model':<25} {'Val Acc':>10}")
    logger.info("=" * 45)
    for name, acc in results.items():
        mark = " << best" if acc == max(results.values()) else ""
        logger.info(f"{name:<25} {acc:>10.4f}{mark}")
    logger.info("=" * 45)

    os.makedirs("./logs", exist_ok=True)
    plot_comparison(results, save_path="./logs/model_comparison.png")
    logger.info("Training pipeline complete.")