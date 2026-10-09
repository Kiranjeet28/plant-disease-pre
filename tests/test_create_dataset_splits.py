import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.create_dataset_splits import (
    leakage_report,
    scan_dataset,
    split_samples,
    validate_manifest,
    write_outputs,
)
from scripts.label_registry import load_registry


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = load_registry(ROOT / "data" / "label_registry.csv", ROOT / "data" / "label_registry_metadata.json")


class CreateDatasetSplitsTests(unittest.TestCase):
    def _make_dataset(self, root: Path, count: int = 12):
        labels = ["Apple___Apple_scab", "Apple___healthy"]
        for index in range(count):
            label = labels[index % 2]
            folder = root / label
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index, 0, 0)).save(folder / f"image-{index}.png")

    def test_reproducibility_ratios_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root)
            samples, issues, stats = scan_dataset(root, "PlantVillage", ENTRIES)
            first, warnings = split_samples(samples, 0.5, 0.25, seed=4)
            second, _ = split_samples(samples, 0.5, 0.25, seed=4)
            self.assertEqual(
                [[sample.image_id for sample in first[name]] for name in first],
                [[sample.image_id for sample in second[name]] for name in second],
            )
            output = Path(temporary) / "manifests"
            reports = Path(temporary) / "reports"
            write_outputs(
                first, issues, warnings, output, reports,
                {"seed": 4, "train_ratio": 0.5, "validation_ratio": 0.25},
                [stats],
            )
            errors, validation_warnings = validate_manifest(
                output / "split_manifest.csv",
                output / "split_metadata.json",
                reports / "data_leakage_report.json",
            )
            self.assertEqual(errors, [])
            self.assertTrue(any("near-duplicate" in warning for warning in validation_warnings))
            self.assertEqual(sum(len(values) for values in first.values()), 12)

    def test_exact_duplicates_stay_together_and_near_duplicates_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self._make_dataset(root, 6)
            duplicate = root / "Apple___healthy" / "duplicate.png"
            duplicate.write_bytes((root / "Apple___healthy" / "image-1.png").read_bytes())
            samples, _, _ = scan_dataset(root, "PlantVillage", ENTRIES)
            splits, _ = split_samples(samples, 0.6, 0.2, seed=2)
            locations = {}
            for partition, values in splits.items():
                for sample in values:
                    locations.setdefault(sample.digest, set()).add(partition)
            self.assertTrue(all(len(partitions) == 1 for partitions in locations.values()))
            rows = [
                {"image_id": sample.image_id, "sha256": sample.digest, "group_id": sample.group_id,
                 "average_hash": sample.average_hash, "partition": partition}
                for partition, values in splits.items() for sample in values
            ]
            report = leakage_report(rows)
            self.assertFalse(report["critical"])

    def test_unresolved_and_corrupt_records_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            (root / "Unknown___label").mkdir(parents=True)
            (root / "Apple___healthy").mkdir(parents=True)
            (root / "Unknown___label" / "unknown.png").write_bytes(b"not an image")
            (root / "Apple___healthy" / "broken.png").write_bytes(b"not an image")
            samples, issues, stats = scan_dataset(root, "PlantVillage", ENTRIES)
            self.assertEqual(samples, [])
            self.assertEqual(len(issues), 2)
            self.assertEqual(stats["discovered_images"], 2)

    def test_empty_dataset_and_deliberate_leakage_fail_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "manifests"
            reports = directory / "reports"
            output.mkdir()
            reports.mkdir()
            with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "image_id", "path", "source_dataset", "source_label", "canonical_id",
                    "crop", "is_healthy", "sha256", "group_id", "average_hash", "partition",
                ])
                writer.writeheader()
                row = {field: "x" for field in writer.fieldnames}
                row.update({"image_id": "same", "partition": "train"})
                writer.writerow(row)
                row["partition"] = "test"
                writer.writerow(row)
            (output / "split_metadata.json").write_text(json.dumps({"configuration": {"seed": 1}}), encoding="utf-8")
            (reports / "data_leakage_report.json").write_text(json.dumps({"critical": True}), encoding="utf-8")
            errors, _ = validate_manifest(
                output / "split_manifest.csv",
                output / "split_metadata.json",
                reports / "data_leakage_report.json",
            )
            self.assertTrue(any("cross partitions" in error for error in errors))
            self.assertTrue(any("critical leakage" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
