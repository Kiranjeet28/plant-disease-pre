"""Validation-only confidence calibration and uncertainty utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "split_manifest.csv"
DEFAULT_OUTPUT = ROOT / "reports"


class CalibrationPrerequisiteError(RuntimeError):
    """Raised when calibration cannot safely proceed."""


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    class_labels: tuple[str, ...]
    method: str = "temperature_scaling"

    def apply(self, logits: np.ndarray) -> np.ndarray:
        values = _validate_logits(logits, len(self.class_labels))
        return softmax(values / self.temperature)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "class_labels": list(self.class_labels),
        }


def save_calibration(calibration: TemperatureCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> TemperatureCalibration:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("method") != "temperature_scaling":
            raise ValueError("Unsupported calibration method.")
        temperature = float(payload["temperature"])
        labels = tuple(payload["class_labels"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise CalibrationPrerequisiteError(f"Invalid calibration artifact: {error}") from error
    if not math.isfinite(temperature) or temperature <= 0 or not labels:
        raise CalibrationPrerequisiteError("Calibration artifact has an invalid temperature or class list.")
    return TemperatureCalibration(temperature, labels)


def _validate_logits(logits: np.ndarray, class_count: int | None = None) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or (class_count is not None and values.shape[1] != class_count):
        raise ValueError("Logits must be a 2-D array with the configured class count.")
    if not np.isfinite(values).all():
        raise ValueError("Logits contain non-finite values.")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite 2-D array.")
    shifted = values - np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    targets: Iterable[int],
    class_labels: Iterable[str],
    minimum_examples_per_class: int = 1,
) -> TemperatureCalibration:
    labels = tuple(class_labels)
    values = _validate_logits(logits, len(labels))
    target_values = np.asarray(list(targets), dtype=np.int64)
    if len(target_values) != len(values):
        raise ValueError("Logits and targets have different lengths.")
    if len(target_values) == 0 or np.any(target_values < 0) or np.any(target_values >= len(labels)):
        raise ValueError("Targets are empty or outside the class range.")
    counts = np.bincount(target_values, minlength=len(labels))
    if np.any(counts < minimum_examples_per_class):
        missing = [labels[index] for index, count in enumerate(counts) if count < minimum_examples_per_class]
        raise CalibrationPrerequisiteError(
            f"Insufficient validation examples for classes: {', '.join(missing)}"
        )

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probabilities = softmax(values / temperature)
        return float(-np.log(np.clip(probabilities[np.arange(len(target_values)), target_values], 1e-15, 1)).mean())

    grid = np.linspace(math.log(0.05), math.log(20.0), 241)
    best = min(grid, key=nll)
    return TemperatureCalibration(math.exp(float(best)), labels)


def calibration_metrics(
    probabilities: np.ndarray,
    targets: Iterable[int],
    class_labels: Iterable[str],
    bins: int = 10,
) -> dict:
    values = np.asarray(probabilities, dtype=np.float64)
    labels = tuple(class_labels)
    target_values = np.asarray(list(targets), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(labels) or len(values) != len(target_values):
        raise ValueError("Probabilities, targets, and class labels are incompatible.")
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-empty.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to one.")
    predictions = np.argmax(values, axis=1)
    confidence = np.max(values, axis=1)
    correct = predictions == target_values
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = (confidence >= bin_edges[index]) & (
            confidence <= bin_edges[index + 1] if index == bins - 1 else confidence < bin_edges[index + 1]
        )
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            ece += count / len(values) * abs(accuracy - mean_confidence)
        else:
            accuracy = None
            mean_confidence = None
        reliability.append({
            "bin": index,
            "lower": float(bin_edges[index]),
            "upper": float(bin_edges[index + 1]),
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })
    one_hot = np.eye(len(labels))[target_values]
    brier = float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))
    nll = float(-np.log(np.clip(values[np.arange(len(values)), target_values], 1e-15, 1)).mean())
    return {
        "sample_count": len(values),
        "correct_count": int(correct.sum()),
        "incorrect_count": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "nll": nll,
        "brier_score": brier,
        "ece": float(ece),
        "bins": bins,
        "labels": list(labels),
        "reliability": reliability,
    }


def select_abstention_threshold(
    probabilities: np.ndarray,
    targets: Iterable[int],
    minimum_coverage: float = 0.5,
) -> dict:
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1].")
    values = np.asarray(probabilities, dtype=np.float64)
    target_values = np.asarray(list(targets), dtype=np.int64)
    if values.ndim != 2 or len(values) != len(target_values) or not len(values):
        raise ValueError("Probabilities and targets must be non-empty and aligned.")
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == target_values
    candidates = sorted(set(confidence.tolist()), reverse=True)
    eligible = []
    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage >= minimum_coverage:
            eligible.append((float(correct[accepted].mean()), coverage, float(threshold)))
    if not eligible:
        raise CalibrationPrerequisiteError("No threshold satisfies the minimum coverage.")
    selective_accuracy, coverage, threshold = max(eligible)
    return {
        "threshold": threshold,
        "minimum_coverage": minimum_coverage,
        "coverage": coverage,
        "accepted_count": int(round(coverage * len(values))),
        "rejected_count": len(values) - int(round(coverage * len(values))),
        "selective_accuracy": selective_accuracy,
        "selection_objective": "maximum validation accuracy subject to minimum coverage",
    }


def apply_abstention(probabilities: np.ndarray, threshold: float) -> list[dict]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite 2-D array.")
    results = []
    for row in values:
        index = int(np.argmax(row))
        confidence = float(row[index])
        results.append({
            "status": "accepted" if confidence >= threshold else "abstain",
            "class_index": index if confidence >= threshold else None,
            "confidence": confidence,
            "reason": None if confidence >= threshold else "below_validation_threshold",
        })
    return results


def inspect_image_quality(path: Path, minimum_size: tuple[int, int] = (32, 32)) -> dict:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return {"status": "rejected", "reason": "missing_or_empty_file"}
        with Image.open(path) as image:
            width, height = image.size
            if width < minimum_size[0] or height < minimum_size[1]:
                return {"status": "rejected", "reason": "below_minimum_dimensions", "dimensions": [width, height]}
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)
            variance = float(grayscale.var())
            return {
                "status": "warning" if variance < 10.0 else "accepted",
                "reason": "possibly_blurry" if variance < 10.0 else None,
                "dimensions": [width, height],
                "variance": variance,
            }
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return {"status": "rejected", "reason": f"invalid_image: {error}"}


def load_validation_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise CalibrationPrerequisiteError(f"Unable to read validation manifest: {error}") from error
    validation = [row for row in rows if row.get("partition") == "validation"]
    if not validation:
        raise CalibrationPrerequisiteError("Validation manifest contains no samples.")
    return validation


def write_unavailable_reports(output: Path, reason: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "calibration_schema_version": 1,
        "status": "unavailable",
        "method": "temperature_scaling",
        "calibration_partition": "validation",
        "metrics_before": None,
        "metrics_after": None,
        "abstention": None,
        "reason": reason,
        "limitations": [
            "No validation predictions or compatible logits are available.",
            "OOD performance is not established without in-distribution and OOD examples.",
        ],
    }
    (output / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "abstention_analysis.json").write_text(json.dumps({"status": "unavailable", "reason": reason}, indent=2) + "\n", encoding="utf-8")
    (output / "ood_evaluation.json").write_text(json.dumps({"status": "unavailable", "reason": reason}, indent=2) + "\n", encoding="utf-8")
    with (output / "reliability_diagram_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "lower", "upper", "count", "accuracy", "mean_confidence"])
        writer.writeheader()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        load_validation_manifest(args.manifest)
    except CalibrationPrerequisiteError as error:
        write_unavailable_reports(args.output_dir, str(error))
        print(json.dumps({"status": "unavailable", "reason": str(error)}, indent=2))
        return 0
    raise SystemExit("Validation predictions/logits are required; no automatic model inference is performed.")


if __name__ == "__main__":
    raise SystemExit(main())
