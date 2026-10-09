import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_datasets import audit_local_dataset, write_reports


class AuditDatasetsTests(unittest.TestCase):
    def test_counts_empty_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Apple___healthy").mkdir()
            (root / "empty").mkdir()
            Image.new("RGB", (4, 3), "green").save(root / "Apple___healthy" / "one.png")
            Image.new("RGB", (4, 3), "red").save(root / "outside.png")
            report = audit_local_dataset("PlantVillage", root)
            self.assertEqual(report["total_images"], 1)
            self.assertEqual(report["class_counts"], {"Apple___healthy": 1})
            self.assertEqual(len(report["label_annotation_inconsistencies"]), 1)

    def test_unreadable_and_exact_duplicate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "class-a"
            root.mkdir(parents=True)
            image = Image.new("RGB", (4, 3), "blue")
            image.save(root / "a.png")
            (root / "b.png").write_bytes((root / "a.png").read_bytes())
            (root / "broken.png").write_bytes(b"not an image")
            report = audit_local_dataset("PlantDoc", root.parent)
            self.assertEqual(report["total_images"], 2)
            self.assertEqual(len(report["exact_duplicate_groups"]), 1)
            self.assertEqual(len(report["integrity_issues"]), 1)

    def test_missing_dataset_and_reports_are_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = audit_local_dataset("PlantVillage", Path(temporary) / "missing")
            self.assertEqual(missing["status"], "not_available")
            output = Path(temporary) / "reports"
            write_reports([missing], output)
            summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["datasets"][0]["status"], "not_available")
            self.assertTrue((output / "per_class_image_counts.csv").read_text(encoding="utf-8").startswith("dataset,"))

    def test_source_images_are_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "class-a"
            root.mkdir(parents=True)
            image_path = root / "image.png"
            Image.new("RGB", (4, 3), "green").save(image_path)
            before = image_path.read_bytes()
            audit_local_dataset("PlantVillage", root.parent)
            self.assertEqual(before, image_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
