import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_models import (
    BenchmarkPrerequisiteError,
    benchmark_latency,
    classification_metrics,
    load_test_manifest,
    top_k_accuracy,
    write_reports,
)


class BenchmarkModelsTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self):
        result = classification_metrics(
            ["a", "a", "b", "c"],
            ["a", "b", "b", "c"],
            ["a", "b", "c"],
        )
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["correct_count"], 3)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(len(result["per_class"]), 3)

    def test_top_two_and_invalid_lengths(self):
        self.assertEqual(top_k_accuracy(["a", "b"], [["b", "a"], ["c", "b"]], 2), 1.0)
        with self.assertRaises(ValueError):
            classification_metrics(["a"], ["a", "b"], ["a", "b"])

    def test_empty_manifest_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            path.write_text(
                "image_id,path,source_dataset,source_label,canonical_id,partition\n",
                encoding="utf-8",
            )
            with self.assertRaises(BenchmarkPrerequisiteError):
                load_test_manifest(path)

    def test_latency_is_repeatable_and_excludes_warmup(self):
        result = benchmark_latency(lambda: 1, iterations=3, warmups=2)
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["warmups"], 2)
        self.assertGreaterEqual(result["median_ms"], 0)
        self.assertFalse(result["includes_preprocessing"])

    def test_json_and_csv_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = {"models": []}
            write_reports(output, summary, [], {"status": "not_measured"})
            self.assertEqual(json.loads((output / "benchmark_summary.json").read_text())["models"], [])
            with (output / "model_comparison.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), ["model", "status", "reason", "sample_count", "accuracy", "macro_f1", "weighted_f1"])


if __name__ == "__main__":
    unittest.main()
