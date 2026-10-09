import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.inference_router import InferenceRouter, ModelCapability, load_capabilities
from scripts.model_adapters import StandardizedPrediction


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeAdapter:
    def __init__(self, prediction=None, error=None):
        self.prediction = prediction
        self.error = error
        self.calls = 0

    def predict(self, data):
        self.calls += 1
        if self.error:
            raise self.error
        return self.prediction


def prediction(**updates):
    values = {
        "model_id": "model-a",
        "raw_scores": (0.8, 0.2),
        "calibrated_probabilities": None,
        "source_class_index": 0,
        "source_label": "source",
        "canonical_id": "apple.apple_scab",
        "display_label": "Apple scab",
        "confidence": 0.8,
        "calibration_status": "not_calibrated",
        "abstained": False,
        "abstention_reason": None,
        "input_warnings": (),
        "compatibility_warnings": (),
        "inference_duration_ms": 1.2,
    }
    values.update(updates)
    return StandardizedPrediction(**values)


class InferenceRouterTests(unittest.TestCase):
    def test_capability_registry_and_deterministic_selection(self):
        capabilities = load_capabilities()
        self.assertEqual(len(capabilities), 3)
        router = InferenceRouter(capabilities)
        selected, reason = router.select_model("Tomato")
        self.assertEqual(selected.model_id, "plantvillage-efficientnetb0-onnx")
        self.assertIn("deterministic", reason)

    def test_successful_routing_and_response_shape(self):
        capability = ModelCapability("model-a", "fake", "ready", ("Apple",), "not_calibrated", "unavailable", "unavailable", ())
        adapter = FakeAdapter(prediction())
        router = InferenceRouter([capability], lambda _: adapter)
        result = router.predict(image_bytes(), crop="Apple")
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.prediction_available)
        self.assertEqual(result.canonical_disease_id, "apple.apple_scab")
        self.assertEqual(adapter.calls, 1)
        self.assertIn("selected_model_id", result.to_dict())

    def test_unavailable_models_are_excluded_and_unknown_crop_is_safe(self):
        capability = ModelCapability("disabled", "fake", "unavailable", ("Apple",), "unavailable", "unavailable", "unavailable", ())
        router = InferenceRouter([capability], lambda _: self.fail("disabled model loaded"))
        result = router.predict(image_bytes(), crop="Tomato")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "no_eligible_model")

    def test_abstention_and_quality_warnings_propagate(self):
        capability = ModelCapability("model-a", "fake", "ready", ("Apple",), "not_calibrated", "unavailable", "unavailable", ())
        adapter = FakeAdapter(prediction(
            abstained=True,
            abstention_reason="below_validation_threshold",
            input_warnings=("possibly_blurry",),
            calibrated_probabilities=(0.7, 0.3),
        ))
        router = InferenceRouter([capability], lambda _: adapter)
        result = router.predict(image_bytes(), crop="Apple")
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(result.abstained)
        self.assertEqual(result.image_quality_warnings, ("possibly_blurry",))
        self.assertEqual(result.error_code, "prediction_abstained")

    def test_invalid_input_and_adapter_failure_are_safe(self):
        capability = ModelCapability("model-a", "fake", "ready", ("Apple",), "unavailable", "unavailable", "unavailable", ())
        adapter = FakeAdapter(error=RuntimeError("failed"))
        router = InferenceRouter([capability], lambda _: adapter)
        self.assertEqual(router.predict(b"").error_code, "invalid_image")
        self.assertEqual(router.predict(image_bytes(), crop="Apple").error_code, "adapter_inference_failed")


if __name__ == "__main__":
    unittest.main()
