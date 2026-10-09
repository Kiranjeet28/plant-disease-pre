import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.label_registry import (
    REQUIRED_COLUMNS,
    RegistryValidationError,
    check_model_labels,
    diseases_for_crop,
    healthy_entries,
    load_registry,
    lookup_source,
    source_labels_for_canonical,
    unresolved_entries,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "label_registry.csv"
METADATA = ROOT / "data" / "label_registry_metadata.json"


def write_fixture(directory: Path, rows: list[dict[str, str]], metadata: dict | None = None):
    registry = directory / "registry.csv"
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata or json.loads(METADATA.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    return registry, metadata_path


def valid_row(**updates):
    row = {
        "canonical_id": "apple.apple_scab",
        "crop": "Apple",
        "disease": "Apple scab",
        "is_healthy": "false",
        "source_dataset": "PlantVillage",
        "source_label": "Apple___Apple_scab",
        "synonyms": "",
        "review_status": "verified",
    }
    row.update(updates)
    return row


class LabelRegistryTests(unittest.TestCase):
    def test_valid_registry_loads_and_supports_healthy_lookup(self):
        entries = load_registry(REGISTRY, METADATA)
        self.assertEqual(len(entries), 38)
        self.assertEqual(lookup_source(entries, "PlantVillage", "Tomato___healthy").canonical_id, "tomato.healthy")
        self.assertTrue(healthy_entries(entries))
        self.assertEqual(len(diseases_for_crop(entries, "Tomato")), 10)

    def test_many_to_one_verified_source_labels_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [
                valid_row(),
                valid_row(
                    source_dataset="PlantDoc",
                    source_label="apple_scab_variant",
                ),
            ]
            registry, metadata = write_fixture(directory, rows)
            self.assertTrue(validate_registry(registry, metadata).is_valid)
            entries = load_registry(registry, metadata)
            self.assertEqual(len(source_labels_for_canonical(entries, "apple.apple_scab")), 2)

    def test_duplicate_ids_and_conflicting_source_mappings_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [
                valid_row(),
                valid_row(),
                valid_row(canonical_id="apple.other", source_label="Apple___Other"),
                valid_row(canonical_id="apple.third", source_label="Apple___Apple_scab"),
            ]
            registry, metadata = write_fixture(directory, rows)
            validation = validate_registry(registry, metadata)
            self.assertFalse(validation.is_valid)
            self.assertTrue(any("duplicate canonical_id/source" in error for error in validation.errors))
            self.assertTrue(any("conflicting mapping" in error for error in validation.errors))

    def test_missing_columns_and_malformed_metadata_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            registry = directory / "registry.csv"
            registry.write_text("canonical_id,crop\nid,Apple\n", encoding="utf-8")
            metadata = directory / "metadata.json"
            metadata.write_text("{}", encoding="utf-8")
            validation = validate_registry(registry, metadata)
            self.assertFalse(validation.is_valid)
            self.assertTrue(any("metadata missing schema" in error for error in validation.errors))
            self.assertTrue(any("missing required columns" in error for error in validation.errors))

    def test_unknown_and_unresolved_labels_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [valid_row(review_status="unresolved", source_label="future_label")]
            registry, metadata = write_fixture(directory, rows)
            entries = load_registry(registry, metadata)
            self.assertIsNone(lookup_source(entries, "PlantVillage", "future_label"))
            self.assertEqual(len(unresolved_entries(entries)), 1)
            result = check_model_labels(entries, "PlantVillage", ["future_label"])
            self.assertEqual(result["unresolved"], ["future_label"])

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            registry, metadata = write_fixture(directory, [valid_row(review_status="pending")])
            with self.assertRaises(RegistryValidationError):
                load_registry(registry, metadata)


if __name__ == "__main__":
    unittest.main()
