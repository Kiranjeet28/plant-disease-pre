import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.calibrate_models import (
    CalibrationPrerequisiteError,
    TemperatureCalibration,
    apply_abstention,
    calibration_metrics,
    fit_temperature,
    inspect_image_quality,
    load_calibration,
    save_calibration,
    select_abstention_threshold,
    softmax,
    write_unavailable_reports,
)


class CalibrateModelsTests(unittest.TestCase):
    def test_temperature_scaling_and_persistence_shape(self):
        logits = np.array([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]])
        calibration = fit_temperature(logits, [0, 1, 0, 1], ["a", "b"])
        calibrated = calibration.apply(logits)
        self.assertEqual(calibrated.shape, (4, 2))
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))
        self.assertEqual(TemperatureCalibration(**{
            "temperature": calibration.temperature,
            "class_labels": calibration.class_labels,
        }).to_dict()["method"], "temperature_scaling")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            save_calibration(calibration, path)
            self.assertEqual(load_calibration(path).temperature, calibration.temperature)

    def test_invalid_and_insufficient_calibration_inputs(self):
        with self.assertRaises(ValueError):
            softmax(np.array([[np.nan, 1.0]]))
        with self.assertRaises(CalibrationPrerequisiteError):
            fit_temperature(np.array([[1.0, 0.0]]), [0], ["a", "b"])

    def test_calibration_metrics_and_threshold(self):
        probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
        metrics = calibration_metrics(probabilities, [0, 1, 1], ["a", "b"], bins=5)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertIn("nll", metrics)
        self.assertEqual(len(metrics["reliability"]), 5)
        threshold = select_abstention_threshold(probabilities, [0, 1, 1], minimum_coverage=2 / 3)
        self.assertGreaterEqual(threshold["coverage"], 2 / 3)
        decisions = apply_abstention(probabilities, threshold["threshold"])
        self.assertIn(decisions[0]["status"], {"accepted", "abstain"})

    def test_quality_checks_distinguish_invalid_small_and_valid_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = inspect_image_quality(root / "missing.png")
            self.assertEqual(missing["status"], "rejected")
            small = root / "small.png"
            Image.new("RGB", (4, 4), "black").save(small)
            self.assertEqual(inspect_image_quality(small)["status"], "rejected")
            valid = root / "valid.png"
            Image.new("RGB", (40, 40), "red").save(valid)
            self.assertIn(inspect_image_quality(valid)["status"], {"warning", "accepted"})

    def test_unavailable_reports_are_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_unavailable_reports(output, "no validation samples")
            summary = json.loads((output / "calibration_summary.json").read_text())
            self.assertEqual(summary["status"], "unavailable")
            self.assertIsNone(summary["metrics_before"])


if __name__ == "__main__":
    unittest.main()
