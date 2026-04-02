# models.py — all model architectures

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from src.utils import get_logger
import tensorflow as tf
logger = get_logger("models")


def build_baseline_cnn(num_classes: int = 4, image_size: tuple = (150, 150)) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(*image_size, 3)),
        layers.Conv2D(32,  (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,  (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    logger.info("Built Baseline CNN")
    return model


def build_mobilenet_tl(num_classes: int = 4, image_size: tuple = (150, 150)) -> models.Sequential:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="mobilenet_transfer")
    logger.info("Built MobileNetV2 Transfer Learning model")
    return model


def build_mobilenet_finetuned(base_model, unfreeze_last: int = 20) -> models.Sequential:
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last]:
        layer.trainable = False
    logger.info(f"Fine-tuned MobileNetV2: last {unfreeze_last} layers unfrozen")
    return base_model


def build_efficientnet(num_classes: int = 4, image_size: tuple = (150, 150)) -> models.Model:
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = tf.keras.layers.Rescaling(scale=255.0)(inputs)  # if generator gives [0,1]

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*image_size, 3)
    )
    base.trainable = False

    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=output, name="efficientnet_b0")
    logger.info("Built EfficientNetB0 Transfer Learning model")
    return model


def build_optuna_cnn(params: dict, num_classes: int = 4, image_size: tuple = (150, 150)) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(*image_size, 3)),
        layers.Conv2D(params["filters_1"], (3,3), activation="relu"),
        BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(params["filters_2"], (3,3), activation="relu"),
        BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(params["filters_3"], (3,3), activation="relu"),
        BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        Dense(params["dense_units"], activation="relu"),
        Dropout(params["dropout"]),
        Dense(num_classes, activation="softmax"),
    ], name="optuna_best_cnn")
    logger.info(f"Built Optuna CNN with params: {params}")
    return model


def compile_model(model, lr: float = 1e-3):
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )
    return model
