"""Validation and lookup interface for the canonical disease registry."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "label_registry.csv"
DEFAULT_METADATA = ROOT / "data" / "label_registry_metadata.json"
REQUIRED_COLUMNS = (
    "canonical_id",
    "crop",
    "disease",
    "is_healthy",
    "source_dataset",
    "source_label",
    "synonyms",
    "review_status",
)
ALLOWED_REVIEW_STATUS = {"verified", "unreviewed", "unresolved"}
ALLOWED_SOURCE_DATASETS = {"PlantVillage", "PlantDoc"}
TRUE_VALUES = {"true"}
FALSE_VALUES = {"false"}


@dataclass(frozen=True)
class RegistryEntry:
    canonical_id: str
    crop: str
    disease: str
    is_healthy: bool
    source_dataset: str
    source_label: str
    synonyms: tuple[str, ...]
    review_status: str


class RegistryValidationError(ValueError):
    """Raised when registry errors prevent safe lookup."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class RegistryValidation:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _parse_synonyms(value: str, row_number: int, warnings: list[str]) -> tuple[str, ...]:
    if not value.strip():
        return ()
    parts = tuple(part.strip() for part in value.split("|"))
    if any(not part for part in parts) or len(set(parts)) != len(parts):
        warnings.append(f"row {row_number}: malformed synonyms")
        return ()
    return parts


def _is_slug(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", value))


def validate_rows(rows: list[dict[str, str]], metadata: dict) -> RegistryValidation:
    errors: list[str] = []
    warnings: list[str] = []
    schema = metadata.get("schema")
    if not isinstance(schema, dict):
        errors.append("metadata missing schema object")
    else:
        metadata_required = schema.get("required_columns")
        if metadata_required != list(REQUIRED_COLUMNS):
            errors.append("metadata required_columns does not match registry schema")
        if set(schema.get("allowed_review_statuses", [])) != ALLOWED_REVIEW_STATUS:
            errors.append("metadata allowed_review_statuses is incomplete or invalid")
        if set(schema.get("allowed_source_datasets", [])) != ALLOWED_SOURCE_DATASETS:
            errors.append("metadata allowed_source_datasets is incomplete or invalid")
    if not isinstance(metadata.get("sources"), dict):
        errors.append("metadata missing sources object")
    if not rows:
        errors.append("registry contains no rows")
        return RegistryValidation(tuple(errors), tuple(warnings))
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing_columns:
        errors.append(f"missing required columns: {', '.join(missing_columns)}")
        return RegistryValidation(tuple(errors), tuple(warnings))
    ids: dict[str, int] = {}
    mappings: dict[tuple[str, str], tuple[str, int]] = {}
    evidence_ids: set[str] = set()
    metadata_sources = metadata.get("sources", {})
    for row_number, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if column == "synonyms":
                continue
            if not row.get(column, "").strip():
                errors.append(f"row {row_number}: empty required field {column}")
        canonical_id = row["canonical_id"].strip()
        if canonical_id and not _is_slug(canonical_id):
            errors.append(f"row {row_number}: malformed canonical_id {canonical_id!r}")
        if canonical_id and canonical_id not in ids:
            ids[canonical_id] = row_number
        status = row["review_status"].strip()
        if status not in ALLOWED_REVIEW_STATUS:
            errors.append(f"row {row_number}: invalid review_status {status!r}")
        healthy = row["is_healthy"].strip().casefold()
        if healthy not in TRUE_VALUES | FALSE_VALUES:
            errors.append(f"row {row_number}: invalid is_healthy {row['is_healthy']!r}")
        if not row["crop"].strip() or not row["disease"].strip():
            errors.append(f"row {row_number}: missing crop or disease identity")
        dataset = row["source_dataset"].strip()
        if dataset not in ALLOWED_SOURCE_DATASETS:
            errors.append(f"row {row_number}: unsupported source_dataset {dataset!r}")
        elif dataset not in metadata_sources:
            errors.append(f"row {row_number}: source dataset lacks metadata evidence: {dataset}")
        mapping_key = (dataset, row["source_label"])
        previous = mappings.get(mapping_key)
        if previous and previous[0] != canonical_id:
            errors.append(
                f"row {row_number}: conflicting mapping for {dataset}/{row['source_label']!r}; "
                f"row {previous[1]} maps to {previous[0]!r}"
            )
        elif previous:
            errors.append(
                f"row {row_number}: duplicate canonical_id/source mapping for "
                f"{dataset}/{row['source_label']!r}"
            )
        else:
            mappings[mapping_key] = (canonical_id, row_number)
        _parse_synonyms(row["synonyms"], row_number, warnings)
        if status == "verified":
            evidence_ids.add(canonical_id)
    if not evidence_ids:
        warnings.append("no verified mappings have documented source evidence")
    for source_name in metadata_sources:
        if source_name in ALLOWED_SOURCE_DATASETS and source_name == "PlantDoc":
            warnings.append("PlantDoc has no local class labels; mappings remain unresolved")
    return RegistryValidation(tuple(errors), tuple(warnings))


def load_registry(
    registry_path: Path = DEFAULT_REGISTRY,
    metadata_path: Path = DEFAULT_METADATA,
) -> list[RegistryEntry]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation = validate_rows(rows, metadata)
    if not validation.is_valid:
        raise RegistryValidationError(list(validation.errors), list(validation.warnings))
    entries: list[RegistryEntry] = []
    for row in rows:
        entries.append(
            RegistryEntry(
                canonical_id=row["canonical_id"],
                crop=row["crop"],
                disease=row["disease"],
                is_healthy=row["is_healthy"].casefold() == "true",
                source_dataset=row["source_dataset"],
                source_label=row["source_label"],
                synonyms=_parse_synonyms(row["synonyms"], 0, []),
                review_status=row["review_status"],
            )
        )
    return entries


def validate_registry(
    registry_path: Path = DEFAULT_REGISTRY,
    metadata_path: Path = DEFAULT_METADATA,
) -> RegistryValidation:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with registry_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
        return RegistryValidation((f"unable to read registry or metadata: {error}",), ())
    return validate_rows(rows, metadata)


def lookup_source(
    entries: Iterable[RegistryEntry], dataset: str, source_label: str
) -> RegistryEntry | None:
    for entry in entries:
        if (
            entry.source_dataset == dataset
            and entry.source_label == source_label
            and entry.review_status == "verified"
        ):
            return entry
    return None


def source_labels_for_canonical(
    entries: Iterable[RegistryEntry], canonical_id: str
) -> list[RegistryEntry]:
    return [entry for entry in entries if entry.canonical_id == canonical_id]


def diseases_for_crop(
    entries: Iterable[RegistryEntry], crop: str, verified_only: bool = True
) -> list[RegistryEntry]:
    return sorted(
        [
            entry
            for entry in entries
            if entry.crop == crop and (not verified_only or entry.review_status == "verified")
        ],
        key=lambda entry: entry.canonical_id,
    )


def healthy_entries(
    entries: Iterable[RegistryEntry], verified_only: bool = True
) -> list[RegistryEntry]:
    return [
        entry
        for entry in entries
        if entry.is_healthy and (not verified_only or entry.review_status == "verified")
    ]


def unresolved_entries(entries: Iterable[RegistryEntry]) -> list[RegistryEntry]:
    return [entry for entry in entries if entry.review_status != "verified"]


def check_model_labels(
    entries: Iterable[RegistryEntry], dataset: str, labels: Iterable[str]
) -> dict[str, list[str]]:
    mapped: list[str] = []
    unresolved: list[str] = []
    for label in labels:
        if lookup_source(entries, dataset, label) is None:
            unresolved.append(label)
        else:
            mapped.append(label)
    return {"mapped": mapped, "unresolved": unresolved}


def main() -> int:
    validation = validate_registry()
    payload = {
        "valid": validation.is_valid,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }
    print(json.dumps(payload, indent=2))
    return 0 if validation.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
