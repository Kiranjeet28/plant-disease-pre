import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.split_dataset import inspect_dataset, split_records


class SplitDatasetTests(unittest.TestCase):
    def test_split_is_reproducible_and_keeps_duplicate_group_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ("class-a", "class-b"):
                folder = root / label
                folder.mkdir()
                for index in range(10):
                    Image.new("RGB", (32, 24), color=(index, 0, 0)).save(
                        folder / f"image-{index}.png"
                    )
                (folder / "duplicate.png").write_bytes(
                    (folder / "image-0.png").read_bytes()
                )
            labels = ["class-a", "class-b"]
            report, records = inspect_dataset(root, labels, {".png"})
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["records"], 22)
            self.assertTrue(report["duplicate_groups"])
            first = split_records(records, 0.7, 0.15, seed=7)
            second = split_records(records, 0.7, 0.15, seed=7)
            self.assertEqual(
                [[record.path.name for record in first[name]] for name in first],
                [[record.path.name for record in second[name]] for name in second],
            )
            locations = {}
            for split, split_records_list in first.items():
                for record in split_records_list:
                    locations.setdefault(record.group, set()).add(split)
            self.assertTrue(all(len(split_names) == 1 for split_names in locations.values()))

    def test_missing_and_extra_classes_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "class-a").mkdir()
            (root / "extra").mkdir()
            report, records = inspect_dataset(root, ["class-a", "class-b"], {".png"})
            self.assertEqual(report["status"], "incompatible_classes")
            self.assertEqual(report["missing_classes"], ["class-b"])
            self.assertEqual(report["extra_classes"], ["extra"])
            self.assertEqual(records, [])

    def test_small_class_cannot_be_represented_in_every_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "class-a"
            folder.mkdir()
            for index in range(2):
                Image.new("RGB", (32, 24), color=(index, 0, 0)).save(
                    folder / f"image-{index}.png"
                )
            _, records = inspect_dataset(root, ["class-a"], {".png"})
            splits = split_records(records, 0.7, 0.15, seed=7)
            present = {
                split for split, split_records_list in splits.items()
                if any(record.label == "class-a" for record in split_records_list)
            }
            self.assertLess(len(present), 3)


if __name__ == "__main__":
    unittest.main()
