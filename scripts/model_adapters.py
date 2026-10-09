"""Model adapters for verified checkpoints.

This module standardizes model calls without changing the production API or
selecting between models. Only the verified ONNX production artifact is
registered by default.
"""

from __future__ import annotations

import io
import json
import hashlib
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
DEFAULT_KERAS_MODEL = ROOT / "model" / "plant_disease_efficientnet.keras"
DEFAULT_PYTORCH_MODEL = ROOT / "model" / "mobilenetv2_plant.pth"
DEFAULT_MOBILE_ONNX_MODEL = ROOT / "model" / "mobilenetv2_plant.onnx"
DEFAULT_CLASS_NAMES = ROOT / "model" / "class_names.txt"
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
    input_layout: str = "NHWC"
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
    raw_output_dtype: str = "unknown"

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
            "raw_output_dtype": self.raw_output_dtype,
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


def keras_efficientnet_config(
    checkpoint: Path = DEFAULT_KERAS_MODEL,
    labels_path: Path = DEFAULT_CLASS_NAMES,
) -> AdapterConfig:
    labels = tuple(
        line.split(": ", 1)[1].strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if ": " in line
    )
    if not labels:
        raise AdapterError(f"Unable to read indexed Keras labels: {labels_path}")
    return AdapterConfig(
        model_id="plantvillage-efficientnetb0-keras",
        architecture="EfficientNetB0",
        checkpoint=checkpoint,
        input_size=(224, 224),
        channels=3,
        color_order="RGB",
        normalization=(
            "raw float32 pixel values in [0, 255]; EfficientNet Keras "
            "preprocessing is built into the model"
        ),
        output_type="probabilities",
        source_dataset="PlantVillage",
        labels=labels,
    )


def mobilenetv2_pytorch_config(
    checkpoint: Path = DEFAULT_PYTORCH_MODEL,
    labels_path: Path = DEFAULT_LABELS,
) -> AdapterConfig:
    return AdapterConfig(
        model_id="mobilenetv2-plant-pth",
        architecture="MobileNetV2",
        checkpoint=checkpoint,
        input_size=(224, 224),
        channels=3,
        color_order="RGB",
        normalization=(
            "ToTensor followed by ImageNet normalization: "
            "mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)"
        ),
        output_type="logits",
        source_dataset="PlantVillage",
        labels=_load_labels(labels_path),
    )


def mobilenetv2_onnx_config(
    checkpoint: Path = DEFAULT_MOBILE_ONNX_MODEL,
    labels_path: Path = DEFAULT_LABELS,
) -> AdapterConfig:
    config = mobilenetv2_pytorch_config(
        checkpoint=checkpoint, labels_path=labels_path
    )
    return AdapterConfig(
        **{
            **config.__dict__,
            "model_id": "mobilenetv2-plant-onnx",
            "input_layout": "NCHW",
        }
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


def _preprocess_nchw(image_bytes: bytes, input_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(input_size, Image.Resampling.BILINEAR)
            values = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise AdapterError(f"Invalid or corrupted input image: {error}") from error
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return np.transpose((values - mean) / std, (2, 0, 1))[np.newaxis, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AdapterError(f"Unable to hash model artifact: {error}") from error
    return digest.hexdigest()


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
            "input_layout": self.config.input_layout,
            "calibration_artifact": (
                str(self.config.calibration_artifact)
                if self.config.calibration_artifact
                else None
            ),
            "artifact_sha256": _file_sha256(self.config.checkpoint),
        }

    def diagnostics(
        self, image_bytes: bytes, prediction: StandardizedPrediction
    ) -> dict[str, Any]:
        processed = self.preprocess(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as image:
            decoded_size = list(image.size)
        return {
            "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "decoded_size": decoded_size,
            "preprocessed_shape": list(processed.shape),
            "preprocessed_dtype": str(processed.dtype),
            "preprocessed_range": [
                float(np.min(processed)),
                float(np.max(processed)),
            ],
            "preprocessed_values": processed.reshape(-1)[:8].astype(float).tolist(),
            "raw_output_shape": [len(prediction.raw_scores)],
            "raw_output_dtype": prediction.raw_output_dtype,
            "raw_output_values": list(prediction.raw_scores[:8]),
            "output_activation": self.config.output_type,
            "source_label_count": len(self.config.labels),
            "source_label_order_sha256": hashlib.sha256(
                "\n".join(self.config.labels).encode("utf-8")
            ).hexdigest(),
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
        expected = (
            [None, self.config.channels, self.config.input_size[0], self.config.input_size[1]]
            if self.config.input_layout == "NCHW"
            else [None, self.config.input_size[0], self.config.input_size[1], self.config.channels]
        )
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
        if self.config.input_layout == "NCHW":
            return _preprocess_nchw(image_bytes, self.config.input_size)
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
                raw_output_dtype=str(raw.dtype),
            )
        abstained = False
        reason = None
        confidence = float(probabilities[index])
        if self.config.abstention_threshold is not None:
            decision = apply_abstention(probabilities[np.newaxis, :], self.config.abstention_threshold)[0]
            abstained = decision["status"] == "abstain"
            reason = decision["reason"]
        return StandardizedPrediction(
            self.config.model_id, tuple(raw.tolist()),
            None if self._calibration is None else tuple(probabilities.tolist()),
            index, source_label, entry.canonical_id, entry.disease, confidence,
            "not_calibrated" if self._calibration is None else "calibrated",
            abstained, reason, warnings, tuple(self._compatibility_warnings), duration,
            raw_output_dtype=str(raw.dtype),
        )


class KerasModelAdapter(ModelAdapter):
    def __init__(
            self,
            config: AdapterConfig | None = None,
            model: Any | None = None,
            registry_path: Path = DEFAULT_REGISTRY,
            registry_metadata_path: Path = DEFAULT_REGISTRY_METADATA,
    ):
            self.config = config or keras_efficientnet_config()
            self._entries = load_registry(registry_path, registry_metadata_path)
            self._model = model
            self._compatibility_warnings: list[str] = []
            self.validate()

    def validate(self) -> None:
            validate_config(self.config, self._entries)
            if self._model is None:
                try:
                    import tensorflow as tf
                    self._model = tf.keras.models.load_model(
                        self.config.checkpoint, compile=False
                    )
                except (ImportError, OSError, ValueError, RuntimeError) as error:
                    raise AdapterError(f"Unable to load Keras checkpoint: {error}") from error
            if tuple(self._model.input_shape[1:]) != (224, 224, 3):
                raise AdapterError(
                    f"Keras input shape {self._model.input_shape} does not match (None, 224, 224, 3)."
                )
            if tuple(self._model.output_shape[1:]) != (len(self.config.labels),):
                raise AdapterError("Keras output width does not match source label count.")
            self._compatibility_warnings.append(
                "No compatible validation-derived calibration artifact is available."
            )

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
            return _preprocess(image_bytes, self.config.input_size)

    def predict(self, image_bytes: bytes) -> StandardizedPrediction:
            quality = tuple(inspect_image_quality_from_bytes(image_bytes))
            input_array = self.preprocess(image_bytes)
            start = time.perf_counter()
            try:
                raw = np.asarray(self._model(input_array, training=False).numpy()[0])
            except (AttributeError, OSError, RuntimeError, ValueError) as error:
                raise AdapterError(f"Keras inference failed: {error}") from error
            duration = (time.perf_counter() - start) * 1000
            if raw.shape != (len(self.config.labels),) or not np.isfinite(raw).all():
                raise AdapterError("Keras output is non-finite or has an incompatible shape.")
            if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=1e-4):
                raise AdapterError("Configured Keras probability output is not normalized.")
            index = int(np.argmax(raw))
            source_label = self.config.labels[index]
            entry = lookup_source(self._entries, self.config.source_dataset, source_label)
            if entry is None:
                return StandardizedPrediction(
                    self.config.model_id, tuple(raw.tolist()), tuple(raw.tolist()),
                    None, source_label, None, None, None, "not_calibrated", True,
                    "unresolved_source_label", quality, tuple(self._compatibility_warnings),
                    duration,
                    raw_output_dtype=str(raw.dtype),
                )
            return StandardizedPrediction(
                self.config.model_id, tuple(raw.tolist()), None, index, source_label,
                entry.canonical_id, entry.disease, float(raw[index]), "not_calibrated",
                False, None, quality, tuple(self._compatibility_warnings), duration,
                raw_output_dtype=str(raw.dtype),
            )


class PytorchMobileNetV2Adapter(ModelAdapter):
    def __init__(
        self,
        config: AdapterConfig | None = None,
        model: Any | None = None,
        registry_path: Path = DEFAULT_REGISTRY,
        registry_metadata_path: Path = DEFAULT_REGISTRY_METADATA,
    ):
        self.config = config or mobilenetv2_pytorch_config()
        self._entries = load_registry(registry_path, registry_metadata_path)
        self._model = model
        self._torch = None
        self._compatibility_warnings: list[str] = []
        self.validate()

    def validate(self) -> None:
        validate_config(self.config, self._entries)
        try:
            import torch
            from torchvision import models
        except (ImportError, OSError, RuntimeError) as error:
            raise AdapterError(f"PyTorch runtime is unavailable: {error}") from error
        self._torch = torch
        if self._model is None:
            try:
                self._model = models.mobilenet_v2(weights=None)
                self._model.classifier[1] = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.2),
                    torch.nn.Linear(self._model.classifier[1].in_features, len(self.config.labels)),
                )
                state = torch.load(
                    self.config.checkpoint, map_location="cpu", weights_only=True
                )
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]
                self._model.load_state_dict(state, strict=True)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                raise AdapterError(
                    f"Unable to load MobileNetV2 checkpoint: {error}"
                ) from error
        output_weight = self._model.classifier[1][1].weight
        if tuple(output_weight.shape) != (len(self.config.labels), 1280):
            raise AdapterError(
                f"MobileNetV2 classifier shape {tuple(output_weight.shape)} "
                f"does not match ({len(self.config.labels)}, 1280)."
            )
        self._model.eval()

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB").resize(
                    self.config.input_size, Image.Resampling.BILINEAR
                )
                values = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise AdapterError(f"Invalid or corrupted input image: {error}") from error
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        normalized = (values - mean) / std
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bytes: bytes) -> StandardizedPrediction:
        quality = tuple(inspect_image_quality_from_bytes(image_bytes))
        input_array = self.preprocess(image_bytes)
        start = time.perf_counter()
        try:
            with self._torch.no_grad():
                raw_tensor = self._model(self._torch.from_numpy(input_array))
            raw = np.asarray(raw_tensor.detach().cpu().numpy()[0])
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            raise AdapterError(f"PyTorch inference failed: {error}") from error
        duration = (time.perf_counter() - start) * 1000
        if raw.shape != (len(self.config.labels),) or not np.isfinite(raw).all():
            raise AdapterError("PyTorch output is non-finite or has an incompatible shape.")
        shifted = raw - np.max(raw)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        source_label = self.config.labels[index]
        entry = lookup_source(self._entries, self.config.source_dataset, source_label)
        if entry is None:
            raise AdapterError(f"Source label lacks a verified mapping: {source_label}")
        return StandardizedPrediction(
            self.config.model_id,
            tuple(raw.tolist()),
            None,
            index,
            source_label,
            entry.canonical_id,
            entry.disease,
            float(probabilities[index]),
            "not_calibrated",
            False,
            None,
            quality,
            tuple(self._compatibility_warnings),
            duration,
            raw_output_dtype=str(raw.dtype),
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
        "plantvillage-efficientnetb0-keras": "ready",
        "mobilenetv2-plant-pth": "ready",
        "mobilenetv2-plant-onnx": "ready",
    }
