import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent / "yakshagana_work" / "saved_models"
)
MAX_CACHE_SIDE = 512


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = MAX_CACHE_SIDE / max(width, height)
    if scale < 1:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )

    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=95)
    decoded = tf.io.decode_jpeg(encoded.getvalue(), channels=3)
    return tf.image.resize(
        tf.cast(decoded, tf.float32), (image_size, image_size)
    ).numpy()


def build_model(class_count: int, image_size: int, seed: int) -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, seed=seed),
            layers.RandomZoom(0.10, seed=seed),
            layers.RandomTranslation(0.05, 0.05, seed=seed),
            layers.RandomContrast(0.20, seed=seed),
            layers.RandomBrightness((-0.4, 0.2), value_range=(0, 255), seed=seed),
        ],
        name="train_augmentation",
    )
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    x = layers.Lambda(mobilenet_v2.preprocess_input, name="preprocess")(x)
    backbone = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)
    outputs = layers.Dense(class_count, activation="softmax", name="vesha")(x)
    return keras.Model(inputs, outputs, name="MobileNetV2_vesha")


class VeshaPredictor:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
        with (artifact_dir / "class_mapping.json").open(encoding="utf-8") as file:
            metadata = json.load(file)

        model_name = metadata.get("best_model")
        if model_name != "MobileNetV2":
            raise ValueError(f"Unsupported model in class mapping: {model_name!r}")

        self.classes = metadata["classes"]
        self.image_size = int(metadata["img_size"])
        self.other_class = metadata.get("other_class")
        if not self.classes:
            raise ValueError("The class mapping contains no classes.")

        weights_path = artifact_dir / f"{model_name}.weights.h5"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        self.model = build_model(
            class_count=len(self.classes),
            image_size=self.image_size,
            seed=int(metadata.get("seed", 42)),
        )
        self.model.load_weights(weights_path)
        self.inference_model = keras.Model(
            self.model.input,
            [
                self.model.output,
                self.model.get_layer("dense_hidden").output,
            ],
            name="vesha_and_embedding",
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess_image(image, self.image_size)

    def predict_many(
        self,
        images: list[Image.Image],
        include_embedding: bool = False,
    ) -> list[dict]:
        if not images:
            return []
        batch = np.stack([self.preprocess(image) for image in images])
        return self.predict_preprocessed(batch, include_embedding=include_embedding)

    def predict_preprocessed(
        self,
        batch: np.ndarray,
        include_embedding: bool = False,
    ) -> list[dict]:
        if batch.size == 0:
            return []
        predictions, embeddings = self.inference_model(batch, training=False)
        predictions = predictions.numpy()
        embeddings = embeddings.numpy()
        embeddings /= np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True),
            1e-12,
        )
        results = []
        for index, scores in enumerate(predictions):
            order = np.argsort(-scores)
            top_index = int(order[0])
            result = {
                "predicted_class": self.classes[top_index],
                "confidence": float(scores[top_index]),
                "class_scores": [
                    {"class": self.classes[int(index)], "score": float(scores[index])}
                    for index in order
                ],
                "not_yakshagana_rejection_supported": self.other_class is not None,
            }
            if include_embedding:
                result["embedding"] = embeddings[index]
            results.append(result)
        return results

    def predict(self, image: Image.Image) -> dict:
        return self.predict_many([image])[0]

    def predict_with_embedding(self, image: Image.Image) -> dict:
        return self.predict_many([image], include_embedding=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a Yakshagana vesha class for one image."
    )
    parser.add_argument("image", type=Path, help="Path to a JPG, PNG, or WEBP image")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing class_mapping.json and model weights",
    )
    args = parser.parse_args()

    with Image.open(args.image) as image:
        result = VeshaPredictor(args.artifacts).predict(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
