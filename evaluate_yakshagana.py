import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from predict_yakshagana import DEFAULT_ARTIFACT_DIR, VeshaPredictor
from yakshagana_augment import blur_distance_views


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Yakshagana dataset"
SPLIT_FILE = DEFAULT_ARTIFACT_DIR / "split.csv"
OUTPUT_FILE = DEFAULT_ARTIFACT_DIR / "evaluation.json"
TARGET_VALIDATION_ACCURACY = 0.95
TARGET_VALIDATION_NOVELTY_ACCEPTANCE = 0.75
MINIMUM_VESHA_CONFIDENCE = 0.85
BATCH_SIZE = 16


def load_split(
    predictor: VeshaPredictor,
    split_name: str,
    split_file: Path = SPLIT_FILE,
    include_blurred: bool = False,
):
    rows = []
    with split_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == split_name:
                rows.append(row)
    if not rows:
        raise ValueError(f"No images found for the {split_name!r} split.")

    labels = []
    predictions = []
    embeddings = []
    for offset in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[offset:offset + BATCH_SIZE]
        batch_images = []
        batch_labels = []
        for row in batch_rows:
            path = (DATASET_DIR / Path(row["relpath"])).resolve()
            if not path.is_relative_to(DATASET_DIR.resolve()):
                raise ValueError(f"Dataset path escapes dataset directory: {row['relpath']}")
            with Image.open(path) as image:
                preprocessed = predictor.preprocess(image)
            batch_images.append(preprocessed)
            batch_labels.append(row["vesha"])
            if include_blurred:
                mild, moderate = blur_distance_views(preprocessed)
                batch_images.extend((mild.numpy(), moderate.numpy()))
                batch_labels.extend((row["vesha"], row["vesha"]))
        batch_predictions = predictor.predict_preprocessed(
            np.stack(batch_images),
            include_embedding=True,
        )
        predictions.extend(batch_predictions)
        embeddings.extend(item["embedding"] for item in batch_predictions)
        labels.extend(batch_labels)

    class_indexes = {label: index for index, label in enumerate(predictor.classes)}
    y_true = np.asarray([class_indexes[label] for label in labels], dtype=np.int64)
    y_scores = np.asarray(
        [
            [score["score"] for score in sorted(
                result["class_scores"],
                key=lambda score: class_indexes[score["class"]],
            )]
            for result in predictions
        ],
        dtype=np.float64,
    )
    return rows, labels, y_true, y_scores, np.asarray(embeddings, dtype=np.float64)


def nearest_training_similarity(
    query_embeddings: np.ndarray,
    training_embeddings: np.ndarray,
) -> np.ndarray:
    similarities = query_embeddings @ training_embeddings.T
    return similarities.max(axis=1)


def choose_uncertainty_thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    top_indexes = order[:, 0]
    top_scores = scores[np.arange(len(scores)), top_indexes]
    correct = top_indexes == y_true
    accepted = top_scores >= MINIMUM_VESHA_CONFIDENCE
    accepted_count = int(accepted.sum())
    if not accepted_count:
        raise ValueError(
            "No validation images meet the configured minimum vesha confidence "
            f"of {MINIMUM_VESHA_CONFIDENCE:.0%}."
        )
    accepted_accuracy = float(correct[accepted].mean())

    return {
        "confidence": MINIMUM_VESHA_CONFIDENCE,
        "margin": 0.0,
        "target_accepted_accuracy": TARGET_VALIDATION_ACCURACY,
        "target_accepted_accuracy_met": accepted_accuracy >= TARGET_VALIDATION_ACCURACY,
        "minimum_vesha_confidence": MINIMUM_VESHA_CONFIDENCE,
        "validation_accepted_count": accepted_count,
        "validation_count": int(len(y_true)),
        "validation_accepted_accuracy": accepted_accuracy,
    }


def main(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    calibrate_on_blur_views: bool = False,
) -> None:
    split_file = artifact_dir / "split.csv"
    output_file = artifact_dir / "evaluation.json"
    predictor = VeshaPredictor(artifact_dir)
    _, _, val_true, val_scores, val_embeddings = load_split(
        predictor,
        "val",
        split_file,
        include_blurred=calibrate_on_blur_views,
    )
    _, _, val_clean_true, val_clean_scores, val_clean_embeddings = load_split(
        predictor,
        "val",
        split_file,
    )
    _, _, val_stress_true, val_stress_scores, val_stress_embeddings = load_split(
        predictor,
        "val",
        split_file,
        include_blurred=True,
    )
    thresholds = choose_uncertainty_thresholds(val_true, val_scores)

    train_rows, _, _, _, train_embeddings = load_split(predictor, "train", split_file)
    test_rows, test_labels, test_true, test_scores, test_embeddings = load_split(
        predictor,
        "test",
        split_file,
    )
    _, _, stress_true, stress_scores, stress_embeddings = load_split(
        predictor,
        "test",
        split_file,
        include_blurred=True,
    )
    val_similarity = nearest_training_similarity(val_embeddings, train_embeddings)
    val_clean_similarity = nearest_training_similarity(
        val_clean_embeddings,
        train_embeddings,
    )
    val_stress_similarity = nearest_training_similarity(
        val_stress_embeddings,
        train_embeddings,
    )
    test_similarity = nearest_training_similarity(test_embeddings, train_embeddings)
    stress_similarity = nearest_training_similarity(stress_embeddings, train_embeddings)
    similarity_threshold = float(
        np.quantile(
            val_similarity,
            1.0 - TARGET_VALIDATION_NOVELTY_ACCEPTANCE,
            method="lower",
        )
    )
    val_accepted = val_similarity >= similarity_threshold
    val_top = val_scores.argmax(axis=1)
    val_margin = np.sort(val_scores, axis=1)[:, -1] - np.sort(val_scores, axis=1)[:, -2]
    val_confidence = val_scores.max(axis=1)
    val_uncertain = ~(
        val_similarity < similarity_threshold
    ) & (
        (val_confidence < thresholds["confidence"])
        | (val_margin < thresholds["margin"])
    )
    val_predictions = np.where(
        val_similarity < similarity_threshold,
        -1,
        np.where(val_uncertain, -2, val_top),
    )
    val_clean_top = val_clean_scores.argmax(axis=1)
    val_clean_margin = (
        np.sort(val_clean_scores, axis=1)[:, -1]
        - np.sort(val_clean_scores, axis=1)[:, -2]
    )
    val_clean_confidence = val_clean_scores.max(axis=1)
    val_clean_rejected = val_clean_similarity < similarity_threshold
    val_clean_uncertain = ~val_clean_rejected & (
        (val_clean_confidence < thresholds["confidence"])
        | (val_clean_margin < thresholds["margin"])
    )
    val_clean_predictions = np.where(
        val_clean_rejected,
        -1,
        np.where(val_clean_uncertain, -2, val_clean_top),
    )
    val_stress_rejected = val_stress_similarity < similarity_threshold
    val_stress_top = val_stress_scores.argmax(axis=1)
    val_stress_margin = (
        np.sort(val_stress_scores, axis=1)[:, -1]
        - np.sort(val_stress_scores, axis=1)[:, -2]
    )
    val_stress_confidence = val_stress_scores.max(axis=1)
    val_stress_uncertain = ~val_stress_rejected & (
        (val_stress_confidence < thresholds["confidence"])
        | (val_stress_margin < thresholds["margin"])
    )
    val_stress_predictions = np.where(
        val_stress_rejected,
        -1,
        np.where(val_stress_uncertain, -2, val_stress_top),
    )

    test_rejected = test_similarity < similarity_threshold

    test_top = test_scores.argmax(axis=1)
    test_margin = np.sort(test_scores, axis=1)[:, -1] - np.sort(test_scores, axis=1)[:, -2]
    test_confidence = test_scores.max(axis=1)
    test_uncertain = ~test_rejected & (
        (test_confidence < thresholds["confidence"])
        | (test_margin < thresholds["margin"])
    )
    app_predictions = np.where(
        test_rejected,
        -1,
        np.where(test_uncertain, -2, test_top),
    )

    stress_rejected = stress_similarity < similarity_threshold
    stress_top = stress_scores.argmax(axis=1)
    stress_margin = np.sort(stress_scores, axis=1)[:, -1] - np.sort(stress_scores, axis=1)[:, -2]
    stress_confidence = stress_scores.max(axis=1)
    stress_uncertain = ~stress_rejected & (
        (stress_confidence < thresholds["confidence"])
        | (stress_margin < thresholds["margin"])
    )
    stress_predictions = np.where(
        stress_rejected,
        -1,
        np.where(stress_uncertain, -2, stress_top),
    )

    labels = predictor.classes
    precision, recall, f1, support = precision_recall_fscore_support(
        test_true,
        app_predictions,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    confusion_values = np.zeros((len(labels), len(labels) + 2), dtype=np.int64)
    for actual, predicted in zip(test_true, app_predictions):
        if predicted >= 0:
            column = int(predicted)
        elif predicted == -2:
            column = len(labels)
        else:
            column = len(labels) + 1
        confusion_values[int(actual), column] += 1
    split_counts = Counter()
    total_counts = Counter()
    with split_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            split_counts[(row["vesha"], row["split"])] += 1
            total_counts[row["vesha"]] += 1

    references = {}
    with split_file.open(newline="", encoding="utf-8") as file:
        train_rows = sorted(
            (row for row in csv.DictReader(file) if row["split"] == "train"),
            key=lambda row: row["relpath"].casefold(),
        )
    for row in train_rows:
        references.setdefault(row["vesha"], row["relpath"])
    if set(references) != set(labels):
        raise ValueError("A training reference image is missing for at least one class.")

    report = {
        "model": "MobileNetV2",
        "evaluation_split": "test",
        "counts": {
            "train": int(sum(count for (label, split), count in split_counts.items() if split == "train")),
            "validation": int(sum(count for (label, split), count in split_counts.items() if split == "val")),
            "test": int(len(test_rows)),
        },
        "accuracy": float((app_predictions == test_true).mean()),
        "classifier_accuracy": float((test_top == test_true).mean()),
        "validation_robustness": {
            "calibrated_on_blur_views": calibrate_on_blur_views,
            "views_per_image": 3 if calibrate_on_blur_views else 1,
            "evaluated_images": int(len(val_true)),
            "classifier_accuracy": float((val_top == val_true).mean()),
            "app_accuracy": float((val_predictions == val_true).mean()),
            "novelty_rejection_count": int((val_similarity < similarity_threshold).sum()),
            "uncertain_count": int(val_uncertain.sum()),
            "accepted_accuracy": float(
                (val_top[val_predictions >= 0] == val_true[val_predictions >= 0]).mean()
            ),
        },
        "validation_clean": {
            "images": int(len(val_clean_true)),
            "classifier_accuracy": float((val_clean_top == val_clean_true).mean()),
            "app_accuracy": float((val_clean_predictions == val_clean_true).mean()),
            "novelty_rejection_count": int(val_clean_rejected.sum()),
            "uncertain_count": int(val_clean_uncertain.sum()),
        },
        "synthetic_blur_stress_validation": {
            "views_per_image": 3,
            "evaluated_images": int(len(val_stress_true)),
            "classifier_accuracy": float((val_stress_top == val_stress_true).mean()),
            "app_accuracy": float((val_stress_predictions == val_stress_true).mean()),
            "novelty_rejection_count": int(val_stress_rejected.sum()),
            "uncertain_count": int(val_stress_uncertain.sum()),
        },
        "uncertain_count": int(test_uncertain.sum()),
        "uncertain_rate": float(test_uncertain.mean()),
        "synthetic_blur_stress_test": {
            "views_per_image": 3,
            "evaluated_images": int(len(stress_true)),
            "classifier_accuracy": float((stress_top == stress_true).mean()),
            "app_accuracy": float((stress_predictions == stress_true).mean()),
            "novelty_rejection_count": int(stress_rejected.sum()),
            "novelty_rejection_rate": float(stress_rejected.mean()),
            "uncertain_count": int(stress_uncertain.sum()),
            "uncertain_rate": float(stress_uncertain.mean()),
        },
        "uncertainty_thresholds": thresholds,
        "novelty_screen": {
            "method": "nearest cosine similarity to normalized training-set dense_hidden embeddings",
            "threshold": similarity_threshold,
            "validation_target_positive_acceptance_rate": TARGET_VALIDATION_NOVELTY_ACCEPTANCE,
            "validation_positive_acceptance_rate": float(val_accepted.mean()),
            "validation_count": int(len(val_similarity)),
            "test_positive_rejection_count": int(test_rejected.sum()),
            "test_positive_rejection_rate": float(test_rejected.mean()),
            "training_embedding_count": int(len(train_rows)),
        },
        "classes": [
            {
                "label": label,
                "image_count": total_counts[label],
                "train_count": split_counts[(label, "train")],
                "validation_count": split_counts[(label, "val")],
                "test_count": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(labels)
        ],
        "confusion_matrix": {
            "actual_labels": labels,
            "predicted_labels": labels + ["Uncertain", "Rejected as Not Yakshagana"],
            "values": confusion_values.tolist(),
        },
        "non_yakshagana": {
            "supported": False,
            "screen_available": True,
            "method": "positive_only_novelty_screen",
            "false_acceptance_rate": None,
            "false_rejection_rate": float(test_rejected.mean()),
            "message": (
                "Best-effort novelty screen calibrated only on Yakshagana validation images. "
                "No non-Yakshagana test images are available, so the false-acceptance rate "
                "is unknown; an unrelated image may still resemble the training embeddings."
            ),
        },
        "references": references,
        "confidence_notice": "Confidence does not guarantee that the prediction is correct.",
    }
    np.save(
        artifact_dir / "novelty_training_embeddings.npy",
        train_embeddings.astype(np.float32),
        allow_pickle=False,
    )
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote metrics and fixed references to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Yakshagana model artifact directory.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing model weights, class mapping, and split manifest",
    )
    parser.add_argument(
        "--blur-robust-validation",
        action="store_true",
        help="Calibrate thresholds on clean plus synthetic blur/distance validation views",
    )
    arguments = parser.parse_args()
    main(arguments.artifacts, arguments.blur_robust_validation)
