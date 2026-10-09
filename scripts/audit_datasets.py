"""Audit local plant-disease datasets without modifying source data.

The auditor intentionally does not download datasets, create splits, or rename
classes. Missing local roots are represented explicitly in the JSON report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "dataset_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt", ".csv"}

SOURCE_METADATA: dict[str, dict[str, Any]] = {
    "PlantVillage": {
        "source": "https://huggingface.co/datasets/mohanty/PlantVillage",
        "source_revision": "9e97599868962bd0079b8db4b7f1efa9185fa1e7",
        "license": "CC BY-SA 3.0",
        "task": "image classification",
        "configurations": ["color", "grayscale", "segmented"],
        "supplied_splits": {"train": 43596, "test": 10709},
        "upstream_total_reported": 54306,
        "upstream_total_in_card_metadata": 54305,
        "upstream_species_reported": 14,
        "upstream_diseases_reported": 26,
        "features": ["image", "image_path", "label", "crop", "disease", "leaf_id"],
    },
    "PlantDoc": {
        "source": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "source_revision": "5467f6012d78d1c446145d5f582da6096f852ae8",
        "license": "CC BY 4.0",
        "task": "cropped image classification",
        "upstream_total_reported": 2598,
        "upstream_species_reported": 13,
        "upstream_diseases_reported": 17,
        "object_detection_repository": (
            "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset"
        ),
        "features": ["image files; classification labels are directory-derived"],
    },
}


def _iter_files(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*")) if path.is_file())


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _average_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((8, 8))
    pixels = [grayscale.getpixel((x, y)) for y in range(8) for x in range(8)]
    average = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= average else "0" for pixel in pixels)


def _class_from_path(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else None


def _annotation_issue(path: Path) -> str | None:
    try:
        if path.suffix.lower() == ".xml":
            root = ET.parse(path).getroot()
            if root.find(".//object") is None and root.find(".//annotation") is None:
                return "XML contains no object or annotation record"
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not value:
                return "JSON annotation is empty"
        elif path.suffix.lower() in {".txt", ".csv"}:
            if not path.read_text(encoding="utf-8", errors="strict").strip():
                return "annotation file is empty"
    except (OSError, UnicodeError, ET.ParseError, json.JSONDecodeError) as error:
        return f"unreadable annotation: {error}"
    return None


def audit_local_dataset(name: str, root: Path | None) -> dict[str, Any]:
    """Return a JSON-serializable audit for a class-folder dataset."""
    result: dict[str, Any] = {
        "dataset_name": name,
        "source_metadata": SOURCE_METADATA[name],
        "local_root": str(root) if root else None,
        "status": "not_available",
        "total_images": 0,
        "class_counts": {},
        "crop_species": [],
        "disease_labels": [],
        "healthy_labels": [],
        "file_formats": {},
        "image_dimensions": {},
        "integrity_issues": [],
        "exact_duplicate_groups": [],
        "potential_near_duplicate_groups": [],
        "label_annotation_inconsistencies": [],
        "folder_structure": {},
        "metadata_schema": {
            "classification_labels": "top-level directory name",
            "annotation_files": {},
        },
    }
    if root is None or not root.exists():
        result["unresolved_questions"] = [
            "Download or provide a local dataset root before local counts can be audited."
        ]
        return result
    if not root.is_dir():
        result["status"] = "invalid_root"
        result["integrity_issues"].append({"path": str(root), "issue": "not a directory"})
        return result

    result["status"] = "ok"
    class_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    digest_paths: defaultdict[str, list[str]] = defaultdict(list)
    ahash_paths: defaultdict[str, list[str]] = defaultdict(list)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    result["folder_structure"] = {
        "top_level_directories": classes,
        "classification_layout": "dataset_root/<class label>/<image files>",
        "top_level_files": sorted(path.name for path in root.iterdir() if path.is_file()),
    }

    for path in _iter_files(root):
        suffix = path.suffix.lower()
        if suffix in ANNOTATION_EXTENSIONS and suffix not in IMAGE_EXTENSIONS:
            issue = _annotation_issue(path)
            annotation = result["metadata_schema"]["annotation_files"]
            annotation[suffix] = annotation.get(suffix, 0) + 1
            if issue:
                result["label_annotation_inconsistencies"].append(
                    {"path": _relative(path, root), "issue": issue}
                )
            continue
        if suffix not in IMAGE_EXTENSIONS:
            result["integrity_issues"].append(
                {"path": _relative(path, root), "issue": "unsupported file extension"}
            )
            continue
        label = _class_from_path(path, root)
        if label is None:
            result["label_annotation_inconsistencies"].append(
                {"path": _relative(path, root), "issue": "missing class label directory"}
            )
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                channels = len(image.getbands())
                perceptual_hash = _average_hash(image)
        except (OSError, UnidentifiedImageError, RuntimeError, ValueError) as error:
            result["integrity_issues"].append(
                {"path": _relative(path, root), "issue": f"unreadable image: {error}"}
            )
            continue
        relative = _relative(path, root)
        class_counts[label] += 1
        format_counts[suffix.lstrip(".")] += 1
        dimensions[f"{width}x{height}"] += 1
        digest_paths[digest].append(relative)
        ahash_paths[perceptual_hash].append(relative)

    result["total_images"] = sum(class_counts.values())
    result["class_counts"] = dict(sorted(class_counts.items()))
    result["file_formats"] = dict(sorted(format_counts.items()))
    result["image_dimensions"] = dict(sorted(dimensions.items()))
    result["crop_species"] = sorted({label.split("___", 1)[0] for label in class_counts})
    result["disease_labels"] = sorted(
        {label.split("___", 1)[1] for label in class_counts if "___" in label}
    )
    result["healthy_labels"] = sorted(
        label for label in class_counts if label.casefold().endswith("healthy")
    )
    result["exact_duplicate_groups"] = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(digest_paths.items())
        if len(paths) > 1
    ]
    result["potential_near_duplicate_groups"] = [
        {"average_hash": digest, "paths": paths, "method": "8x8 average hash candidate"}
        for digest, paths in sorted(ahash_paths.items())
        if len(paths) > 1
        and not any(set(paths).issubset(set(group["paths"])) for group in result["exact_duplicate_groups"])
    ]
    result["unresolved_questions"] = [
        "Review perceptual-hash candidates manually; they are not asserted to be duplicates.",
        "Confirm whether any annotation files belong to this classification dataset.",
    ]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(audits: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "audit_schema_version": 1,
        "datasets": audits,
        "project_compatibility": {
            "framework": "TensorFlow/Keras",
            "model_classes": 38,
            "model_input": [224, 224, 3],
            "model_labels_file": "app/labels.json",
            "observation": "PlantVillage class names align with the existing 38-label model; PlantDoc labels must not be remapped or merged in Step 2.",
        },
        "unresolved_questions": [
            question
            for audit in audits
            for question in audit.get("unresolved_questions", [])
        ],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    class_rows = [
        {"dataset": audit["dataset_name"], "class": label, "image_count": count}
        for audit in audits
        for label, count in audit["class_counts"].items()
    ]
    _write_csv(output_dir / "per_class_image_counts.csv", class_rows, ["dataset", "class", "image_count"])
    issue_rows = [
        {"dataset": audit["dataset_name"], "category": "integrity", **issue}
        for audit in audits
        for issue in audit["integrity_issues"]
    ] + [
        {"dataset": audit["dataset_name"], "category": "label_or_annotation", **issue}
        for audit in audits
        for issue in audit["label_annotation_inconsistencies"]
    ]
    _write_csv(output_dir / "integrity_and_label_issues.csv", issue_rows, ["dataset", "category", "path", "issue"])
    duplicate_rows = [
        {
            "dataset": audit["dataset_name"],
            "kind": kind,
            "key": group.get("sha256", group.get("average_hash")),
            "paths": json.dumps(group["paths"]),
        }
        for audit in audits
        for kind, groups in (
            ("exact", audit["exact_duplicate_groups"]),
            ("potential_near", audit["potential_near_duplicate_groups"]),
        )
        for group in groups
    ]
    _write_csv(output_dir / "duplicate_findings.csv", duplicate_rows, ["dataset", "kind", "key", "paths"])
    for audit in audits:
        (output_dir / f"{audit['dataset_name'].lower()}_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantvillage-root", type=Path, default=None)
    parser.add_argument("--plantdoc-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audits = [
        audit_local_dataset("PlantVillage", args.plantvillage_root),
        audit_local_dataset("PlantDoc", args.plantdoc_root),
    ]
    write_reports(audits, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
