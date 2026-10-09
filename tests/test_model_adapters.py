import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.model_adapters import (
    AdapterError,
    KerasModelAdapter,
    OnnxModelAdapter,
    available_adapters,
    mobilenetv2_pytorch_config,
    production_onnx_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeValue:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [FakeValue("input", [None, 224, 224, 3])]

    def get_outputs(self):
        return [FakeValue("output", [None, 38])]

    def run(self, names, inputs):
        return [self.output]


def image_bytes(size=(40, 40), color=(20, 40, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class ModelAdaptersTests(unittest.TestCase):
    def test_verified_onnx_adapter_loads_real_artifact(self):
        adapter = OnnxModelAdapter()
        self.assertEqual(adapter.metadata()["input_size"], [224, 224])
        self.assertEqual(len(adapter.config.labels), 38)
        result = adapter.predict(image_bytes())
        self.assertEqual(result.model_id, "plantvillage-efficientnetb0-onnx")
        self.assertEqual(len(result.raw_scores), 38)
        self.assertIn(result.calibration_status, {"not_calibrated", "calibrated"})
        self.assertIsNotNone(result.canonical_id)

    def test_preprocessing_and_standardized_output(self):
        output = np.full((1, 38), 1 / 38, dtype=np.float32)
        adapter = OnnxModelAdapter(session=FakeSession(output))
        processed = adapter.preprocess(image_bytes())
        self.assertEqual(processed.shape, (1, 224, 224, 3))
        result = adapter.predict(image_bytes())
        payload = result.to_dict()
        self.assertEqual(payload["source_class_index"], 0)
        self.assertEqual(len(payload["raw_scores"]), 38)
        self.assertIn("abstained", payload)

    def test_diagnostics_identify_input_and_artifact(self):
        output = np.full((1, 38), 1 / 38, dtype=np.float32)
        adapter = OnnxModelAdapter(session=FakeSession(output))
        result = adapter.predict(image_bytes())
        diagnostics = adapter.diagnostics(image_bytes(), result)
        self.assertEqual(diagnostics["input_sha256"], hashlib.sha256(image_bytes()).hexdigest())
        self.assertEqual(diagnostics["preprocessed_shape"], [1, 224, 224, 3])
        self.assertEqual(diagnostics["raw_output_dtype"], "float32")
        self.assertEqual(len(adapter.metadata()["artifact_sha256"]), 64)

    def test_onnx_index_16_maps_to_peach_bacterial_spot(self):
        output = np.full((1, 38), 0.001, dtype=np.float32)
        output[0, 16] = 0.963
        output /= output.sum()
        adapter = OnnxModelAdapter(session=FakeSession(output))
        result = adapter.predict(image_bytes())
        self.assertEqual(result.source_label, "Peach___Bacterial_spot")
        self.assertEqual(result.canonical_id, "peach.bacterial_spot")
        self.assertEqual(result.display_label, "Bacterial spot")

    def test_invalid_output_and_missing_checkpoint_fail_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = production_onnx_config(Path(temporary) / "missing.onnx")
            with self.assertRaises(AdapterError):
                OnnxModelAdapter(config=config, session=FakeSession(np.full((1, 38), 1 / 38)))
        adapter = OnnxModelAdapter(session=FakeSession(np.full((1, 2), 0.5)))
        with self.assertRaises(AdapterError):
            adapter.predict(image_bytes())

    def test_unresolved_mapping_is_not_remapped(self):
        output = np.full((1, 38), 1 / 38, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            labels = json.loads((ROOT / "app" / "labels.json").read_text())
            labels[0] = "unverified_label"
            labels_path = Path(temporary) / "labels.json"
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            config = production_onnx_config(labels_path=labels_path)
            with self.assertRaises(AdapterError):
                OnnxModelAdapter(config=config, session=FakeSession(output))

    def test_ready_and_unavailable_adapter_statuses_are_explicit(self):
        statuses = available_adapters()
        self.assertEqual(statuses["plantvillage-efficientnetb0-onnx"], "ready")
        self.assertEqual(statuses["plantvillage-efficientnetb0-keras"], "ready")
        self.assertEqual(statuses["mobilenetv2-plant-pth"], "ready")

    def test_pytorch_config_has_verified_architecture_and_preprocessing(self):
        config = mobilenetv2_pytorch_config()
        self.assertEqual(config.architecture, "MobileNetV2")
        self.assertEqual(config.input_size, (224, 224))
        self.assertEqual(config.output_type, "logits")
        self.assertEqual(len(config.labels), 38)
        self.assertIn("ImageNet normalization", config.normalization)

    def test_verified_keras_adapter_loads_and_runs_real_checkpoint(self):
        adapter = KerasModelAdapter()
        result = adapter.predict(image_bytes())
        self.assertEqual(result.model_id, "plantvillage-efficientnetb0-keras")
        self.assertEqual(len(result.raw_scores), 38)
        self.assertIsNotNone(result.canonical_id)
        self.assertEqual(result.calibration_status, "not_calibrated")


if __name__ == "__main__":
    unittest.main()
