import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.train_models import (
    TrainingPrerequisiteError,
    build_model,
    load_manifest,
    train_experiment,
)


class TrainModelsTests(unittest.TestCase):
    def test_model_construction_shapes(self):
        model = build_model("small_cnn", 3, image_size=(32, 32))
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 3))

    def test_empty_verified_manifest_blocks_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            metadata = root / "metadata.json"
            manifest.write_text(
                "image_id,path,source_dataset,source_label,canonical_id,crop,is_healthy,sha256,group_id,average_hash,partition\n",
                encoding="utf-8",
            )
            metadata.write_text(json.dumps({"eligible_samples": 0}), encoding="utf-8")
            with self.assertRaises(TrainingPrerequisiteError):
                train_experiment(manifest, metadata, root / "experiments")

    def test_missing_manifest_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "image_id", "path", "source_dataset", "source_label", "canonical_id",
                    "crop", "is_healthy", "sha256", "group_id", "partition",
                ])
                writer.writeheader()
                writer.writerow({
                    "image_id": "one", "path": str(root / "missing.png"),
                    "source_dataset": "PlantVillage", "source_label": "x",
                    "canonical_id": "x", "crop": "x", "is_healthy": "false",
                    "sha256": "x", "group_id": "x", "partition": "train",
                })
            with self.assertRaises(TrainingPrerequisiteError):
                load_manifest(manifest)

    def test_synthetic_cpu_training_saves_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_paths = []
            for index, partition in enumerate(("train", "train", "validation", "validation")):
                path = root / f"{index}.png"
                Image.new("RGB", (32, 32), (index * 40, 0, 0)).save(path)
                image_paths.append((path, partition))
            manifest = root / "manifest.csv"
            fields = ["image_id", "path", "source_dataset", "source_label", "canonical_id",
                      "crop", "is_healthy", "sha256", "group_id", "average_hash", "partition"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, (path, partition) in enumerate(image_paths):
                    writer.writerow({
                        "image_id": str(index), "path": str(path),
                        "source_dataset": "PlantVillage", "source_label": "label",
                        "canonical_id": f"class_{index % 2}", "crop": "Test",
                        "is_healthy": "false", "sha256": str(index),
                        "group_id": str(index), "average_hash": str(index),
                        "partition": partition,
                    })
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"eligible_samples": 4}), encoding="utf-8")
            experiment = train_experiment(
                manifest, metadata, root / "experiments",
                architecture="small_cnn", epochs=1, batch_size=2, seed=3,
            )
            self.assertTrue((experiment / "best.keras").exists())
            details = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(details["status"], "completed")
            self.assertEqual(details["epochs_completed"], 1)


if __name__ == "__main__":
    unittest.main()
