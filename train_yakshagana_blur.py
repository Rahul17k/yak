import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from predict_yakshagana import DEFAULT_ARTIFACT_DIR, build_model, preprocess_image
from yakshagana_augment import blur_distance_views, random_blur_distance


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Yakshagana dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "blur_robust_candidate"
BATCH_SIZE = 8
SEED = 42
FINE_TUNE_LAYERS = 30
EPOCHS = 8


def load_images(
    split: str,
    classes: list[str],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    class_indexes = {label: index for index, label in enumerate(classes)}
    split_file = DEFAULT_ARTIFACT_DIR / "split.csv"
    with split_file.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]

    if not rows:
        raise ValueError(f"No images found for the {split!r} split.")
    for row in rows:
        image_path = (DATASET_DIR / Path(row["relpath"])).resolve()
        if not image_path.is_relative_to(DATASET_DIR.resolve()):
            raise ValueError(f"Dataset path escapes dataset directory: {row['relpath']}")
        with Image.open(image_path) as image:
            images.append(preprocess_image(image, image_size))
        labels.append(class_indexes[row["vesha"]])
    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int32)


def make_training_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
) -> tf.data.Dataset:
    sample_weights = class_weights[labels]
    dataset = tf.data.Dataset.from_tensor_slices((images, labels, sample_weights))
    dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)

    def augment(image, label, sample_weight):
        return random_blur_distance(image), label, sample_weight

    return (
        dataset.map(augment, num_parallel_calls=1)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )


def make_validation_dataset(
    images: np.ndarray,
    labels: np.ndarray,
) -> tf.data.Dataset:
    validation_images = []
    validation_labels = []
    for image, label in zip(images, labels):
        mild, moderate = blur_distance_views(image)
        validation_images.extend((image, mild.numpy(), moderate.numpy()))
        validation_labels.extend((label, label, label))
    dataset = tf.data.Dataset.from_tensor_slices(
        (np.asarray(validation_images, dtype=np.float32), np.asarray(validation_labels, dtype=np.int32))
    )
    return dataset.batch(BATCH_SIZE).prefetch(1)


def unfreeze_last_layers(model: keras.Model) -> None:
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2")
    ]
    if len(backbones) != 1:
        raise ValueError(f"Expected one MobileNetV2 backbone, found {len(backbones)}.")
    backbone = backbones[0]
    backbone.trainable = True
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Yakshagana model with synthetic blur and distance augmentation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Separate artifact directory for the candidate model",
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_mapping.json", "split.csv"):
        shutil.copy2(DEFAULT_ARTIFACT_DIR / name, args.output_dir / name)

    with (DEFAULT_ARTIFACT_DIR / "class_mapping.json").open(encoding="utf-8") as file:
        class_mapping = json.load(file)
    classes = class_mapping["classes"]
    image_size = int(class_mapping["img_size"])
    train_images, train_labels = load_images("train", classes, image_size)
    validation_images, validation_labels = load_images("val", classes, image_size)
    present_classes = np.unique(train_labels)
    class_weights = np.ones(len(classes), dtype=np.float32)
    class_weights[present_classes] = compute_class_weight(
        "balanced",
        classes=present_classes,
        y=train_labels,
    )

    model = build_model(len(classes), image_size, SEED)
    initial_weights = DEFAULT_ARTIFACT_DIR / "MobileNetV2.weights.h5"
    if not initial_weights.is_file():
        raise FileNotFoundError(f"Existing model weights not found: {initial_weights}")
    model.load_weights(initial_weights)
    unfreeze_last_layers(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    started = time.time()
    history = model.fit(
        make_training_dataset(train_images, train_labels, class_weights),
        validation_data=make_validation_dataset(validation_images, validation_labels),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    weights_path = args.output_dir / "MobileNetV2.weights.h5"
    model.save_weights(weights_path)
    metadata = {
        "model": "MobileNetV2",
        "initial_weights": str(initial_weights),
        "training_method": "fine-tuning with synthetic zoom-out, downsampling, and Gaussian blur",
        "train_images": int(len(train_labels)),
        "validation_images": int(len(validation_labels)),
        "epochs": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "image_size": image_size,
        "seed": SEED,
        "train_time_sec": time.time() - started,
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    (args.output_dir / "MobileNetV2_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Candidate weights saved to {weights_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
