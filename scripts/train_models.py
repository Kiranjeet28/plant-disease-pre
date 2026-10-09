"""Safe Step 5 training utilities for verified split manifests.

This module never creates splits and refuses to train when the manifest has no
eligible samples. Training is intentionally separate from the production API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "split_manifest.csv"
DEFAULT_METADATA = ROOT / "data" / "manifests" / "split_metadata.json"
DEFAULT_REGISTRY = ROOT / "data" / "label_registry_metadata.json"
DEFAULT_OUTPUT = ROOT / "experiments"
PARTITIONS = ("train", "validation", "test")


class TrainingPrerequisiteError(RuntimeError):
    """Raised when verified training inputs are unavailable or inconsistent."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {
        "image_id", "path", "source_dataset", "source_label", "canonical_id",
        "crop", "is_healthy", "sha256", "group_id", "partition",
    }
    missing = required - set(fields)
    if missing:
        raise TrainingPrerequisiteError(
            f"Split manifest is missing required fields: {', '.join(sorted(missing))}"
        )
    if not rows:
        raise TrainingPrerequisiteError("Split manifest contains no eligible samples.")
    invalid = [row["partition"] for row in rows if row["partition"] not in PARTITIONS]
    if invalid:
        raise TrainingPrerequisiteError(f"Manifest contains invalid partitions: {sorted(set(invalid))}")
    missing_paths = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing_paths:
        raise TrainingPrerequisiteError(
            f"Manifest references missing image files: {missing_paths[:3]}"
        )
    return rows


def ordered_labels(rows: list[dict[str, str]]) -> list[str]:
    labels = sorted({row["canonical_id"] for row in rows})
    if not labels:
        raise TrainingPrerequisiteError("Manifest contains no canonical labels.")
    return labels


def verify_manifest_metadata(metadata_path: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingPrerequisiteError(f"Unable to read split metadata: {error}") from error
    if metadata.get("eligible_samples", 0) == 0:
        raise TrainingPrerequisiteError("Split metadata records zero eligible samples; training is blocked.")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata["manifest_sha256"] = digest
    return metadata


def build_model(
    architecture: str,
    class_count: int,
    image_size: tuple[int, int] = (224, 224),
    pretrained: bool = False,
) -> tf.keras.Model:
    if class_count < 2:
        raise ValueError("At least two classes are required.")
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    if architecture == "small_cnn":
        x = tf.keras.layers.Rescaling(1 / 255.0)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif architecture == "efficientnet_b0":
        weights = "imagenet" if pretrained else None
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=weights, input_shape=(*image_size, 3)
        )
        backbone.trainable = False
        x = backbone(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs, name=architecture)


def _dataset(rows: list[dict[str, str]], labels: list[str], image_size: tuple[int, int], batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    label_to_index = {label: index for index, label in enumerate(labels)}
    paths = [row["path"] for row in rows]
    targets = [label_to_index[row["canonical_id"]] for row in rows]
    if not paths:
        raise TrainingPrerequisiteError("Requested training partition is empty.")

    def load(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, method="bilinear")
        return tf.cast(image, tf.float32), tf.one_hot(target, len(labels))

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_experiment(
    manifest_path: Path = DEFAULT_MANIFEST,
    split_metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_OUTPUT,
    architecture: str = "small_cnn",
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 20261009,
    pretrained: bool = False,
) -> Path:
    tf.keras.utils.set_random_seed(seed)
    rows = load_manifest(manifest_path)
    split_metadata = verify_manifest_metadata(split_metadata_path, manifest_path)
    labels = ordered_labels(rows)
    if any(not Path(row["path"]).is_file() for row in rows):
        raise TrainingPrerequisiteError("Manifest references missing image files.")
    train_rows = [row for row in rows if row["partition"] == "train"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    if not validation_rows:
        raise TrainingPrerequisiteError("Validation partition is empty; training is blocked.")
    model = build_model(architecture, len(labels), pretrained=pretrained)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    experiment_id = f"{architecture}-{time.strftime('%Y%m%d-%H%M%S')}"
    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "pretrained": pretrained,
        "pretrained_source": "ImageNet via Keras" if pretrained else None,
        "labels": labels,
        "manifest_sha256": split_metadata["manifest_sha256"],
        "registry_metadata": str(DEFAULT_REGISTRY),
        "seed": seed,
        "image_size": [224, 224],
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "augmentation": "none; Step 5 baseline infrastructure",
        "tensorflow_version": tf.__version__,
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "platform": platform.platform(),
        "status": "started",
    }
    (experiment_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    checkpoint = experiment_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(experiment_dir / "history.csv")),
    ]
    history = model.fit(
        _dataset(train_rows, labels, (224, 224), batch_size, True, seed),
        validation_data=_dataset(validation_rows, labels, (224, 224), batch_size, False, seed),
        epochs=epochs,
        callbacks=callbacks,
    )
    metadata.update({"status": "completed", "epochs_completed": len(history.history.get("loss", []))})
    (experiment_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return experiment_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-metadata", dest="split_metadata_path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--architecture", choices=("small_cnn", "efficientnet_b0"), default="small_cnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    try:
        directory = train_experiment(**vars(args))
    except (OSError, ValueError, TrainingPrerequisiteError, tf.errors.InvalidArgumentError) as error:
        parser.error(str(error))
    print(json.dumps({"experiment_dir": str(directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
