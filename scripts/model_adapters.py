"""Model adapters for verified checkpoints.

This module standardizes model calls without changing the production API or
selecting between models. Only the verified ONNX production artifact is
registered by default.
"""

from __future__ import annotations

import io
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from scripts.calibrate_models import apply_abstention, inspect_image_quality, load_calibration
from scripts.label_registry import RegistryEntry, lookup_source, load_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "plant_disease.onnx"
DEFAULT_LABELS = ROOT / "app" / "labels.json"
DEFAULT_REGISTRY = ROOT / "data" / "label_registry.csv"
DEFAULT_REGISTRY_METADATA = ROOT / "data" / "label_registry_metadata.json"
DEFAULT_CALIBRATION = ROOT / "reports" / "calibration" / "production_onnx.json"


class AdapterError(RuntimeError):
    """Raised when an adapter is unavailable or receives incompatible data."""


class InferenceSession(Protocol):
    def get_inputs(self) -> list[Any]: ...
    def get_outputs(self) -> list[Any]: ...
    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class AdapterConfig:
    model_id: str
    architecture: str
    checkpoint: Path
    input_size: tuple[int, int]
    channels: int
    color_order: str
    normalization: str
    output_type: str
    source_dataset: str
    labels: tuple[str, ...]
    calibration_artifact: Path | None = None
    abstention_threshold: float | None = None


@dataclass(frozen=True)
class StandardizedPrediction:
    model_id: str
    raw_scores: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...] | None
    source_class_index: int | None
    source_label: str | None
    canonical_id: str | None
    display_label: str | None
    confidence: float | None
    calibration_status: str
    abstained: bool
    abstention_reason: str | None
    input_warnings: tuple[str, ...] = ()
    compatibility_warnings: tuple[str, ...] = ()
    inference_duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "raw_scores": list(self.raw_scores),
            "calibrated_probabilities": (
                list(self.calibrated_probabilities)
                if self.calibrated_probabilities is not None
                else None
            ),
            "source_class_index": self.source_class_index,
            "source_label": self.source_label,
            "canonical_id": self.canonical_id,
            "display_label": self.display_label,
            "confidence": self.confidence,
            "calibration_status": self.calibration_status,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "input_warnings": list(self.input_warnings),
            "compatibility_warnings": list(self.compatibility_warnings),
            "inference_duration_ms": self.inference_duration_ms,
        }


def _load_labels(path: Path) -> tuple[str, ...]:
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"Unable to read source labels: {error}") from error
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
        raise AdapterError("Source labels must be a non-empty JSON string list.")
    if len(set(labels)) != len(labels):
        raise AdapterError("Source labels contain duplicates.")
    return tuple(labels)


def production_onnx_config(
    checkpoint: Path = DEFAULT_MODEL, labels_path: Path = DEFAULT_LABELS
) -> AdapterConfig:
    labels = _load_labels(labels_path)
    return AdapterConfig(
        model_id="plantvillage-efficientnetb0-onnx",
        architecture="EfficientNetB0",
        checkpoint=checkpoint,
        input_size=(224, 224),
        channels=3,
        color_order="RGB",
        normalization="raw float32 pixel values in [0, 255], matching app/main.py",
        output_type="probabilities",
        source_dataset="PlantVillage",
        labels=labels,
        calibration_artifact=DEFAULT_CALIBRATION,
    )


def validate_config(config: AdapterConfig, registry_entries: list[RegistryEntry]) -> None:
    if not config.checkpoint.is_file():
        raise AdapterError(f"Checkpoint does not exist: {config.checkpoint}")
    if config.input_size[0] <= 0 or config.input_size[1] <= 0 or config.channels != 3:
        raise AdapterError("Invalid model input configuration.")
    if config.output_type not in {"probabilities", "logits"}:
        raise AdapterError(f"Unsupported output type: {config.output_type}")
    unmapped = [
        label for label in config.labels
        if lookup_source(registry_entries, config.source_dataset, label) is None
    ]
    if unmapped:
        raise AdapterError(f"Source labels lack verified registry mappings: {unmapped[:3]}")


def _preprocess(image_bytes: bytes, input_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(input_size, Image.Resampling.NEAREST)
            values = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise AdapterError(f"Invalid or corrupted input image: {error}") from error
    return np.expand_dims(values, axis=0)


class ModelAdapter(ABC):
    config: AdapterConfig

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_bytes: bytes) -> StandardizedPrediction:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "architecture": self.config.architecture,
            "checkpoint": str(self.config.checkpoint),
            "input_size": list(self.config.input_size),
            "channels": self.config.channels,
            "color_order": self.config.color_order,
            "normalization": self.config.normalization,
            "output_type": self.config.output_type,
            "source_dataset": self.config.source_dataset,
            "source_labels": list(self.config.labels),
            "calibration_artifact": (
                str(self.config.calibration_artifact)
                if self.config.calibration_artifact
                else None
            ),
        }


class OnnxModelAdapter(ModelAdapter):
    def __init__(
        self,
        config: AdapterConfig | None = None,
        session: InferenceSession | None = None,
        registry_path: Path = DEFAULT_REGISTRY,
        registry_metadata_path: Path = DEFAULT_REGISTRY_METADATA,
    ):
        self.config = config or production_onnx_config()
        self._entries = load_registry(registry_path, registry_metadata_path)
        self._session = session
        self._input_name: str | None = None
        self._calibration = None
        self._compatibility_warnings: list[str] = []
        self.validate()

    def validate(self) -> None:
        validate_config(self.config, self._entries)
        if self._session is None:
            try:
                self._session = ort.InferenceSession(
                    str(self.config.checkpoint),
                    providers=["CPUExecutionProvider"],
                )
            except (OSError, RuntimeError) as error:
                raise AdapterError(f"Unable to load ONNX checkpoint: {error}") from error
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise AdapterError("ONNX adapter requires exactly one input and one output.")
        input_shape = inputs[0].shape
        expected = [None, self.config.input_size[0], self.config.input_size[1], self.config.channels]
        if list(input_shape[1:]) != expected[1:]:
            raise AdapterError(f"ONNX input shape {input_shape} does not match {expected}.")
        output_shape = outputs[0].shape
        if output_shape[-1] != len(self.config.labels):
            raise AdapterError("ONNX output width does not match source label count.")
        self._input_name = inputs[0].name
        if self.config.calibration_artifact and self.config.calibration_artifact.is_file():
            try:
                self._calibration = load_calibration(self.config.calibration_artifact)
                if tuple(self._calibration.class_labels) != self.config.labels:
                    raise AdapterError("Calibration class ordering does not match adapter labels.")
                self._compatibility_warnings.append(
                    "Calibration artifact is logits-based and cannot be applied to probability outputs."
                )
                self._calibration = None
            except AdapterError:
                raise
            except Exception as error:
                self._compatibility_warnings.append(f"Calibration unavailable: {error}")
        else:
            self._compatibility_warnings.append("No compatible calibration artifact is available.")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        return _preprocess(image_bytes, self.config.input_size)

    def _scores(self, values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values)
        if scores.shape != (len(self.config.labels),) or not np.isfinite(scores).all():
            raise AdapterError("Model output is non-finite or has an incompatible shape.")
        if self.config.output_type == "probabilities":
            if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-4):
                raise AdapterError("Configured probability output is not a normalized finite distribution.")
            return scores.astype(np.float64)
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def predict(self, image_bytes: bytes) -> StandardizedPrediction:
        quality = inspect_image_quality_from_bytes(image_bytes)
        warnings = tuple(quality)
        input_array = self.preprocess(image_bytes)
        start = time.perf_counter()
        try:
            raw = np.asarray(self._session.run(None, {self._input_name: input_array})[0][0])
        except (OSError, RuntimeError, ValueError) as error:
            raise AdapterError(f"ONNX inference failed: {error}") from error
        duration = (time.perf_counter() - start) * 1000
        probabilities = self._scores(raw)
        index = int(np.argmax(probabilities))
        source_label = self.config.labels[index]
        entry = lookup_source(self._entries, self.config.source_dataset, source_label)
        if entry is None:
            return StandardizedPrediction(
                self.config.model_id, tuple(raw.tolist()), tuple(probabilities.tolist()),
                None, source_label, None, None, None, "not_calibrated", True,
                "unresolved_source_label", warnings, tuple(self._compatibility_warnings), duration,
            )
        abstained = False
        reason = None
        confidence = float(probabilities[index])
        if self.config.abstention_threshold is not None:
            decision = apply_abstention(probabilities[np.newaxis, :], self.config.abstention_threshold)[0]
            abstained = decision["status"] == "abstain"
            reason = decision["reason"]
        return StandardizedPrediction(
            self.config.model_id, tuple(raw.tolist()), None if self._calibration is None else tuple(probabilities.tolist()),
            index, source_label, entry.canonical_id, entry.disease, confidence,
            "not_calibrated" if self._calibration is None else "calibrated",
            abstained, reason, warnings, tuple(self._compatibility_warnings), duration,
        )


def inspect_image_quality_from_bytes(image_bytes: bytes) -> list[str]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width < 32 or height < 32:
                return ["below_minimum_dimensions"]
            if np.asarray(image.convert("L"), dtype=np.float32).var() < 10:
                return ["possibly_blurry"]
    except (OSError, UnidentifiedImageError, ValueError):
        return ["invalid_image"]
    return []


def available_adapters() -> dict[str, str]:
    return {
        "plantvillage-efficientnetb0-onnx": "ready",
        "plantvillage-efficientnetb0-keras": "requires_review: no machine-readable adapter metadata",
        "mobilenetv2-plant-pth": "unavailable: framework and class mapping metadata are not verified",
    }
