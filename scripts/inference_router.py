"""Deterministic, safety-preserving inference routing for verified adapters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from scripts.model_adapters import (
    AdapterError,
    ModelAdapter,
    StandardizedPrediction,
    OnnxModelAdapter,
    KerasModelAdapter,
    mobilenetv2_onnx_config,
    PytorchMobileNetV2Adapter,
)
from scripts.label_registry import load_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPABILITIES = ROOT / "data" / "model_capabilities.json"
LOGGER = logging.getLogger(__name__)


class AdapterFactory(Protocol):
    def __call__(self, model_id: str) -> ModelAdapter: ...


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    adapter_type: str
    status: str
    supported_crops: tuple[str, ...]
    calibration_status: str
    abstention_status: str
    benchmark_status: str
    limitations: tuple[str, ...]
    priority: int = 100


@dataclass(frozen=True)
class RouterResponse:
    status: str
    prediction_available: bool
    selected_model_id: str | None
    canonical_disease_id: str | None
    display_label: str | None
    is_healthy: bool | None
    confidence: float | None
    calibration_status: str | None
    abstained: bool
    abstention_reason: str | None
    ood_warning: str | None
    image_quality_warnings: tuple[str, ...]
    supported_crop_information: tuple[str, ...]
    model_selection_reason: str
    inference_time_ms: float | None
    warnings: tuple[str, ...]
    error_code: str | None
    user_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prediction_available": self.prediction_available,
            "selected_model_id": self.selected_model_id,
            "canonical_disease_id": self.canonical_disease_id,
            "display_label": self.display_label,
            "is_healthy": self.is_healthy,
            "confidence": self.confidence,
            "calibration_status": self.calibration_status,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "ood_warning": self.ood_warning,
            "image_quality_warnings": list(self.image_quality_warnings),
            "supported_crop_information": list(self.supported_crop_information),
            "model_selection_reason": self.model_selection_reason,
            "inference_time_ms": self.inference_time_ms,
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "user_message": self.user_message,
        }


@dataclass(frozen=True)
class ModelAnalysis:
    model_id: str
    adapter_type: str
    status: str
    supported_crops: tuple[str, ...]
    prediction: dict[str, Any] | None
    exclusion_reason: str | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter_type": self.adapter_type,
            "status": self.status,
            "supported_crops": list(self.supported_crops),
            "prediction": self.prediction,
            "exclusion_reason": self.exclusion_reason,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class AnalysisResponse:
    request_id: str
    image_validation: dict[str, Any]
    models: tuple[ModelAnalysis, ...]
    final_decision: dict[str, Any]
    benchmark: dict[str, Any]
    comparison: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "image_validation": self.image_validation,
            "models": [model.to_dict() for model in self.models],
            "final_decision": self.final_decision,
            "benchmark": self.benchmark,
            "comparison": self.comparison,
        }


def load_capabilities(path: Path = DEFAULT_CAPABILITIES) -> list[ModelCapability]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload["models"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise AdapterError(f"Unable to read model capability registry: {error}") from error
    capabilities = []
    for item in models:
        required = ("model_id", "adapter_type", "status", "supported_crops")
        if any(field not in item for field in required):
            raise AdapterError(f"Capability entry is missing required fields: {item}")
        capabilities.append(
            ModelCapability(
                model_id=item["model_id"],
                adapter_type=item["adapter_type"],
                status=item["status"],
                supported_crops=tuple(item["supported_crops"]),
                calibration_status=item.get("calibration_status", "unavailable"),
                abstention_status=item.get("abstention_status", "unavailable"),
                benchmark_status=item.get("benchmark_status", "unavailable"),
                limitations=tuple(item.get("limitations", [])),
                priority=int(item.get("priority", 100)),
            )
        )
    return capabilities


class InferenceRouter:
    """Select one verified adapter; never combines model predictions."""

    def __init__(
        self,
        capabilities: Iterable[ModelCapability] | None = None,
        adapter_factory: AdapterFactory | None = None,
        registry_path: Path | None = None,
        registry_metadata_path: Path | None = None,
    ):
        self.capabilities = list(capabilities or load_capabilities())
        self._adapter_factory = adapter_factory or self._default_factory
        self._adapters: dict[str, ModelAdapter] = {}
        self._registry_entries = load_registry(
            registry_path or ROOT / "data" / "label_registry.csv",
            registry_metadata_path or ROOT / "data" / "label_registry_metadata.json",
        )

    @staticmethod
    def _default_factory(model_id: str) -> ModelAdapter:
        if model_id == "plantvillage-efficientnetb0-onnx":
            return OnnxModelAdapter()
        if model_id == "plantvillage-efficientnetb0-keras":
            return KerasModelAdapter()
        if model_id == "mobilenetv2-plant-pth":
            return PytorchMobileNetV2Adapter()
        if model_id == "mobilenetv2-plant-onnx":
            return OnnxModelAdapter(config=mobilenetv2_onnx_config())
        raise AdapterError(f"No verified adapter factory for model {model_id!r}.")

    def _eligible(self, crop: str | None) -> list[ModelCapability]:
        return [
            capability
            for capability in self.capabilities
            if capability.status == "ready"
            and (crop is None or crop in capability.supported_crops)
        ]

    def _get_adapter(self, model_id: str) -> ModelAdapter:
        if model_id not in self._adapters:
            self._adapters[model_id] = self._adapter_factory(model_id)
        return self._adapters[model_id]

    def select_model(self, crop: str | None = None) -> tuple[ModelCapability | None, str]:
        candidates = self._eligible(crop)
        if not candidates:
            return None, "No ready model supports the requested crop or unknown crop."
        # Stable model_id ordering is the documented tie-breaker. Benchmark data
        # is currently unavailable and therefore cannot influence selection.
        selected = sorted(candidates, key=lambda item: (item.priority, item.model_id))[0]
        reason = "Selected highest-priority ready model by configured priority; model_id is the deterministic tie-breaker."
        if selected.benchmark_status == "unavailable":
            reason += " Benchmark metrics are unavailable."
        return selected, reason

    def predict(self, image_bytes: bytes, crop: str | None = None) -> RouterResponse:
        if not image_bytes:
            return self._failure("invalid_image", "No image was provided.", crop)
        capability, reason = self.select_model(crop)
        if capability is None:
            return self._failure("no_eligible_model", "No verified model is available for this input.", crop, reason)
        try:
            adapter = self._get_adapter(capability.model_id)
            prediction = adapter.predict(image_bytes)
            LOGGER.info(
                "model_inference",
                extra={
                    "model_id": capability.model_id,
                    "artifact_sha256": adapter.metadata()["artifact_sha256"],
                    "preprocessing": {
                        "input_size": getattr(adapter.config, "input_size", None),
                        "color_order": getattr(adapter.config, "color_order", None),
                        "normalization": getattr(adapter.config, "normalization", None),
                    },
                    "status": "ok",
                    "inference_duration_ms": prediction.inference_duration_ms,
                },
            )
        except (AdapterError, OSError, ValueError, RuntimeError) as error:
            LOGGER.warning("Inference failed for model %s: %s", capability.model_id, error)
            return self._failure("adapter_inference_failed", "The image could not be safely classified.", crop, reason, capability.model_id)
        return self._standardize(prediction, capability, reason)

    def analyze(
        self,
        image_bytes: bytes,
        crop: str | None = None,
        request_id: str = "",
    ) -> AnalysisResponse:
        """Return observable per-model evidence without changing routing."""
        if not image_bytes:
            return AnalysisResponse(
                request_id,
                {"valid": False, "reason": "empty_upload"},
                tuple(),
                {"status": "unavailable", "error_code": "invalid_image"},
                self._benchmark_metadata(),
                {"compatible": False, "reason": "No image was provided.", "models": []},
            )
        selected, reason = self.select_model(crop)
        analyses: list[ModelAnalysis] = []
        selected_prediction: RouterResponse | None = None
        for capability in self.capabilities:
            if capability.status != "ready":
                analyses.append(ModelAnalysis(
                    capability.model_id, capability.adapter_type, capability.status,
                    capability.supported_crops, None,
                    "Model is not marked ready in the capability registry.",
                    capability.limitations,
                ))
                continue
            if crop is not None and crop not in capability.supported_crops:
                analyses.append(ModelAnalysis(
                    capability.model_id, capability.adapter_type, capability.status,
                    capability.supported_crops, None,
                    f"Model does not support crop {crop!r}.", capability.limitations,
                ))
                continue
            try:
                adapter = self._get_adapter(capability.model_id)
                prediction = adapter.predict(image_bytes)
                LOGGER.info(
                    "model_inference",
                    extra={
                        "model_id": capability.model_id,
                        "artifact_sha256": adapter.metadata()["artifact_sha256"],
                        "preprocessing": {
                            "input_size": getattr(adapter.config, "input_size", None),
                            "color_order": getattr(adapter.config, "color_order", None),
                            "normalization": getattr(adapter.config, "normalization", None),
                        },
                        "status": "ok",
                        "inference_duration_ms": prediction.inference_duration_ms,
                    },
                )
                selected_result = self._standardize(prediction, capability, reason)
                if capability.model_id == (selected.model_id if selected else None):
                    selected_prediction = selected_result
                analyses.append(ModelAnalysis(
                    capability.model_id, capability.adapter_type, capability.status,
                    capability.supported_crops, {
                        **prediction.to_dict(),
                        "score_type": self._score_type(prediction, adapter),
                        "top_predictions": self._top_predictions(
                            prediction, adapter.config.labels
                        ),
                        "diagnostics": {
                            **adapter.diagnostics(image_bytes, prediction),
                            "artifact": adapter.metadata(),
                        },
                    }, None, capability.limitations,
                ))
            except (AdapterError, OSError, ValueError, RuntimeError) as error:
                LOGGER.warning("Analysis failed for model %s: %s", capability.model_id, error)
                analyses.append(ModelAnalysis(
                    capability.model_id, capability.adapter_type, capability.status,
                    capability.supported_crops, None,
                    "Model failed during analysis; see server logs for diagnostics.",
                    capability.limitations,
                ))
        if selected_prediction is None:
            selected_prediction = self._failure(
                "no_eligible_model" if selected is None else "adapter_inference_failed",
                "No reliable model result is available.",
                crop, reason,
                selected.model_id if selected else None,
            )
        return AnalysisResponse(
            request_id,
            {"valid": True, "crop": crop, "quality_warnings": list(
                selected_prediction.image_quality_warnings
            )},
            tuple(analyses),
            {
                **selected_prediction.to_dict(),
                "explanation": selected_prediction.model_selection_reason,
            },
            self._benchmark_metadata(),
            self._compare_predictions(analyses),
        )

    @staticmethod
    def _compare_predictions(analyses: list[ModelAnalysis]) -> dict[str, Any]:
        evaluated = [
            model for model in analyses
            if model.prediction is not None
        ]
        if len(evaluated) < 2:
            return {
                "compatible": False,
                "reason": "At least two evaluated models are required.",
                "models": [model.model_id for model in evaluated],
            }
        predictions = [model.prediction for model in evaluated]
        score_types = {prediction.get("score_type") for prediction in predictions}
        label_orders = {
            tuple(prediction["diagnostics"]["artifact"]["source_labels"])
            for prediction in predictions
        }
        if len(score_types) != 1 or len(label_orders) != 1:
            return {
                "compatible": False,
                "reason": "Class ordering and score semantics are not verified as compatible.",
                "models": [model.model_id for model in evaluated],
            }
        scores = [
            prediction.get("calibrated_probabilities") or prediction["raw_scores"]
            for prediction in predictions
        ]
        deltas = [
            abs(float(left) - float(right))
            for left, right in zip(scores[0], scores[1])
        ]
        return {
            "compatible": True,
            "reason": "Models use the same verified class order and probability semantics.",
            "models": [model.model_id for model in evaluated],
            "top_prediction_matches": len({
                prediction.get("canonical_id") for prediction in predictions
            }) == 1,
            "max_absolute_score_difference": max(deltas),
            "mean_absolute_score_difference": sum(deltas) / len(deltas),
        }

    @staticmethod
    def _score_type(prediction: StandardizedPrediction, adapter: ModelAdapter) -> str:
        if prediction.calibrated_probabilities is not None:
            return (
                "calibrated_probability"
                if prediction.calibration_status == "calibrated"
                else "uncalibrated_probability"
            )
        return adapter.config.output_type

    @staticmethod
    def _top_predictions(
        prediction: StandardizedPrediction, labels: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        scores = prediction.calibrated_probabilities or prediction.raw_scores
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:5]
        return [
            {
                "rank": rank,
                "source_class_index": index,
                "source_label": labels[index],
                "score": float(score),
            }
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _benchmark_metadata() -> dict[str, Any]:
        return {
            "status": "unavailable",
            "evaluation_partition": "test",
            "reason": "Verified test manifest contains no samples.",
            "models": [],
        }

    def _standardize(
        self,
        prediction: StandardizedPrediction,
        capability: ModelCapability,
        reason: str,
    ) -> RouterResponse:
        warnings = tuple(prediction.compatibility_warnings)
        warnings += capability.limitations
        quality = tuple(prediction.input_warnings)
        if prediction.abstained or prediction.canonical_id is None:
            return RouterResponse(
                "uncertain", False, capability.model_id, prediction.canonical_id,
                prediction.display_label, self._health_status(prediction.canonical_id),
                prediction.confidence,
                prediction.calibration_status, True,
                prediction.abstention_reason or "unresolved_canonical_mapping",
                "input_quality_warning" if quality else None, quality, capability.supported_crops,
                reason, prediction.inference_duration_ms, warnings, "prediction_abstained",
                "The model could not provide a reliable disease classification.",
            )
        return RouterResponse(
            "ok", True, capability.model_id, prediction.canonical_id,
            prediction.display_label, self._health_status(prediction.canonical_id),
            prediction.confidence,
            prediction.calibration_status, False, None,
            "input_quality_warning" if quality else None, quality,
            capability.supported_crops, reason, prediction.inference_duration_ms,
            warnings, None, "Prediction generated by a verified model.",
        )

    def _health_status(self, canonical_id: str | None) -> bool | None:
        if canonical_id is None:
            return None
        matches = [entry for entry in self._registry_entries if entry.canonical_id == canonical_id]
        return matches[0].is_healthy if len(matches) == 1 else None

    @staticmethod
    def _failure(
        error_code: str,
        message: str,
        crop: str | None,
        reason: str = "",
        model_id: str | None = None,
    ) -> RouterResponse:
        return RouterResponse(
            "unavailable", False, model_id, None, None, None, None, None, True,
            error_code, None, (), (), reason or "Routing could not select a verified model.",
            None, (), error_code, message,
        )
