# data_loader.py — dataset loading and augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import get_logger

logger = get_logger("data_loader")


def get_data_generators(cfg: dict):
    data_cfg = cfg["data"]
    aug_cfg  = cfg["augmentation"]

    image_size  = tuple(data_cfg["image_size"])
    batch_size  = data_cfg["batch_size"]
    val_split   = data_cfg["validation_split"]
    train_dir   = data_cfg["train_dir"]
    test_dir    = data_cfg["test_dir"]

    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=aug_cfg["rotation_range"],
        width_shift_range=aug_cfg["width_shift_range"],
        height_shift_range=aug_cfg["height_shift_range"],
        zoom_range=aug_cfg["zoom_range"],
        horizontal_flip=aug_cfg["horizontal_flip"],
        brightness_range=aug_cfg["brightness_range"],
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=cfg["project"]["seed"],
        shuffle=True
    )

    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=cfg["project"]["seed"],
        shuffle=False
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    logger.info(f"Train samples : {train_data.samples}")
    logger.info(f"Val samples   : {val_data.samples}")
    logger.info(f"Test samples  : {test_data.samples}")
    logger.info(f"Classes       : {train_data.class_indices}")

    return train_data, val_data, test_data