"""Benchmark available classifiers without training or changing model artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "split_manifest.csv"
DEFAULT_REPORT_DIR = ROOT / "reports"
PARTITIONS = {"train", "validation", "test"}


class BenchmarkPrerequisiteError(RuntimeError):
    """Raised when evaluation cannot be performed safely."""


def load_test_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"image_id", "path", "canonical_id", "partition", "source_dataset", "source_label"}
    missing = sorted(required - set(fields))
    if missing:
        raise BenchmarkPrerequisiteError(f"Manifest missing required fields: {', '.join(missing)}")
    test_rows = [row for row in rows if row["partition"] == "test"]
    if not test_rows:
        raise BenchmarkPrerequisiteError("Verified test manifest contains no samples.")
    if any(row["partition"] not in PARTITIONS for row in rows):
        raise BenchmarkPrerequisiteError("Manifest contains an invalid partition.")
    return test_rows


def classification_metrics(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
    labels: Iterable[str],
) -> dict:
    actual = list(true_labels)
    predicted = list(predicted_labels)
    class_labels = list(labels)
    if len(actual) != len(predicted):
        raise ValueError("Ground-truth and prediction lengths differ.")
    if not class_labels:
        raise ValueError("At least one evaluation label is required.")
    index = {label: position for position, label in enumerate(class_labels)}
    if any(label not in index for label in actual + predicted):
        raise ValueError("Evaluation contains a label outside the declared class order.")
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for truth, prediction in zip(actual, predicted):
        matrix[index[truth], index[prediction]] += 1
    precision, recall, f1, support = [], [], [], []
    for position in range(len(class_labels)):
        true_positive = matrix[position, position]
        false_positive = matrix[:, position].sum() - true_positive
        false_negative = matrix[position, :].sum() - true_positive
        class_support = matrix[position, :].sum()
        p = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        r = true_positive / class_support if class_support else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
        support.append(int(class_support))
    total = len(actual)
    return {
        "sample_count": total,
        "correct_count": int(np.trace(matrix)),
        "incorrect_count": total - int(np.trace(matrix)),
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_f1": float(statistics.mean(f1)) if f1 else None,
        "weighted_f1": float(np.average(f1, weights=support)) if total else None,
        "labels": class_labels,
        "per_class": [
            {
                "label": label,
                "precision": precision[position],
                "recall": recall[position],
                "f1": f1[position],
                "support": support[position],
            }
            for position, label in enumerate(class_labels)
        ],
        "confusion_matrix": matrix.tolist(),
    }


def top_k_accuracy(
    true_labels: Iterable[str], ranked_predictions: Iterable[Iterable[str]], k: int = 2
) -> float | None:
    actual = list(true_labels)
    ranked = [list(predictions) for predictions in ranked_predictions]
    if len(actual) != len(ranked):
        raise ValueError("Ground-truth and ranked prediction lengths differ.")
    if not actual:
        return None
    return sum(truth in predictions[:k] for truth, predictions in zip(actual, ranked)) / len(actual)


def benchmark_latency(
    predict: Callable[[], object], iterations: int = 20, warmups: int = 3
) -> dict:
    if iterations <= 0 or warmups < 0:
        raise ValueError("iterations must be positive and warmups cannot be negative.")
    for _ in range(warmups):
        predict()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        predict()
        durations.append((time.perf_counter() - start) * 1000)
    ordered = sorted(durations)
    return {
        "iterations": iterations,
        "warmups": warmups,
        "median_ms": statistics.median(durations),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "throughput_images_per_second": 1000 / statistics.mean(durations),
        "includes_preprocessing": False,
    }


def _read_image(path: str, image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize(image_size, Image.Resampling.NEAREST), dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise BenchmarkPrerequisiteError(f"Unable to read evaluation image {path}: {error}") from error


def write_reports(
    report_dir: Path,
    summary: dict,
    per_class: list[dict],
    latency: dict,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "latency_benchmark.json").write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "label", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_class)
    with (report_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "status", "reason", "sample_count", "accuracy", "macro_f1", "weighted_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: model.get(field) for field in fields}
            for model in summary.get("models", [])
        )


def run(args: argparse.Namespace) -> int:
    summary = {
        "benchmark_schema_version": 1,
        "evaluation_partition": "test",
        "manifest": str(args.manifest),
        "models": [],
        "limitations": [],
    }
    latency = {"status": "not_measured", "reason": "No model was benchmarked."}
    try:
        rows = load_test_manifest(args.manifest)
        summary["test_sample_count"] = len(rows)
    except (OSError, BenchmarkPrerequisiteError) as error:
        summary["test_sample_count"] = 0
        summary["limitations"].append(str(error))
        summary["models"].append({
            "model": "production_onnx_and_candidates",
            "status": "skipped",
            "reason": str(error),
            "sample_count": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        })
    write_reports(args.report_dir, summary, [], latency)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
