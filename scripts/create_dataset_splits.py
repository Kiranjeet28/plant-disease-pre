"""Create and validate read-only, group-aware dataset split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from PIL import Image, UnidentifiedImageError

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.label_registry import DEFAULT_METADATA, DEFAULT_REGISTRY, lookup_source, load_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = ROOT / "data" / "manifests"
DEFAULT_REPORT_DIR = ROOT / "reports"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    source_dataset: str
    source_label: str
    canonical_id: str
    crop: str
    is_healthy: bool
    digest: str
    group_id: str
    average_hash: str


def _average_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((8, 8))
    return "".join(
        "1" if gray.getpixel((x, y)) >= 128 else "0"
        for y in range(8)
        for x in range(8)
    )


def _image_files(root: Path) -> Iterable[Path]:
    return (
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def scan_dataset(
    root: Path,
    dataset: str,
    entries: list,
) -> tuple[list[Sample], list[dict], dict]:
    """Scan a class-folder root without modifying it."""
    issues: list[dict] = []
    stats = {"status": "ok", "root": str(root), "discovered_images": 0}
    if not root.exists():
        stats.update({"status": "not_available", "error": "dataset root does not exist"})
        return [], issues, stats
    if not root.is_dir():
        stats.update({"status": "invalid_root", "error": "dataset root is not a directory"})
        return [], issues, stats
    samples: list[Sample] = []
    digest_paths: defaultdict[str, list[Path]] = defaultdict(list)
    for path in _image_files(root):
        stats["discovered_images"] += 1
        relative = str(path.relative_to(root))
        parts = path.relative_to(root).parts
        source_label = parts[0] if len(parts) > 1 else ""
        entry = lookup_source(entries, dataset, source_label)
        if entry is None:
            issues.append({"path": relative, "issue": "unresolved source label", "source_label": source_label})
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                average_hash = _average_hash(image)
        except (OSError, UnidentifiedImageError, RuntimeError, ValueError) as error:
            issues.append({"path": relative, "issue": f"unreadable image: {error}"})
            continue
        digest_paths[digest].append(path)
        samples.append(
            Sample(
                image_id=f"{dataset}:{relative}",
                path=path,
                source_dataset=dataset,
                source_label=source_label,
                canonical_id=entry.canonical_id,
                crop=entry.crop,
                is_healthy=entry.is_healthy,
                digest=digest,
                group_id=f"hash:{digest}",
                average_hash=average_hash,
            )
        )
    return samples, issues, stats


def _assign_groups(samples: list[Sample], ratios: dict[str, float], seed: int) -> dict[str, str]:
    groups: defaultdict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    group_items = list(groups.values())
    Random(seed).shuffle(group_items)
    targets = {name: len(samples) * ratio for name, ratio in ratios.items()}
    counts = Counter()
    assignments: dict[str, str] = {}
    for group in group_items:
        # Prefer the partition furthest below its global target, then stable name order.
        partition = max(
            ratios,
            key=lambda name: (targets[name] - counts[name], -list(ratios).index(name)),
        )
        for sample in group:
            assignments[sample.image_id] = partition
            counts[partition] += 1
    return assignments


def split_samples(
    samples: list[Sample],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 20261009,
) -> tuple[dict[str, list[Sample]], list[str]]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must be positive and leave a test partition")
    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": 1 - train_ratio - validation_ratio,
    }
    assignments = _assign_groups(samples, ratios, seed)
    splits = {name: [] for name in ratios}
    for sample in samples:
        splits[assignments[sample.image_id]].append(sample)
    warnings = []
    for canonical_id, total in Counter(s.canonical_id for s in samples).items():
        if total < 3:
            warnings.append(f"{canonical_id}: only {total} eligible sample(s); all partitions cannot be represented")
    return splits, warnings


MANIFEST_FIELDS = [
    "image_id", "path", "source_dataset", "source_label", "canonical_id",
    "crop", "is_healthy", "sha256", "group_id", "average_hash", "partition",
]


def write_outputs(
    splits: dict[str, list[Sample]],
    issues: list[dict],
    warnings: list[str],
    output_dir: Path,
    report_dir: Path,
    config: dict,
    dataset_stats: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for partition, partition_samples in splits.items():
        for sample in partition_samples:
            rows.append({
                "image_id": sample.image_id,
                "path": str(sample.path),
                "source_dataset": sample.source_dataset,
                "source_label": sample.source_label,
                "canonical_id": sample.canonical_id,
                "crop": sample.crop,
                "is_healthy": str(sample.is_healthy).lower(),
                "sha256": sample.digest,
                "group_id": sample.group_id,
                "average_hash": sample.average_hash,
                "partition": partition,
            })
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    excluded = output_dir / "excluded_samples.csv"
    with excluded.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "issue", "source_label"])
        writer.writeheader()
        writer.writerows(issues)
    metadata = {
        "split_schema_version": 1,
        "configuration": config,
        "datasets": dataset_stats,
        "eligible_samples": len(rows),
        "excluded_samples": len(issues),
        "limitations": warnings,
        "source_images_modified": False,
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    class_rows = []
    for partition, partition_samples in splits.items():
        for canonical_id, count in sorted(Counter(s.canonical_id for s in partition_samples).items()):
            class_rows.append({"partition": partition, "canonical_id": canonical_id, "image_count": count})
    with (report_dir / "split_class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["partition", "canonical_id", "image_count"])
        writer.writeheader()
        writer.writerows(class_rows)
    leakage = leakage_report(rows)
    (report_dir / "data_leakage_report.json").write_text(json.dumps(leakage, indent=2) + "\n", encoding="utf-8")
    summary = {
        "configuration": config,
        "eligible_samples": len(rows),
        "partition_counts": {name: len(values) for name, values in splits.items()},
        "excluded_samples": len(issues),
        "unresolved_or_corrupt_records": issues,
        "rare_class_warnings": warnings,
        "dataset_stats": dataset_stats,
        "leakage": leakage,
    }
    (report_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def leakage_report(rows: list[dict]) -> dict:
    by_digest: defaultdict[str, set[str]] = defaultdict(set)
    by_group: defaultdict[str, set[str]] = defaultdict(set)
    by_average_hash: defaultdict[str, set[str]] = defaultdict(set)
    by_image: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_digest[row["sha256"]].add(row["partition"])
        by_group[row["group_id"]].add(row["partition"])
        by_average_hash[row["average_hash"]].add(row["partition"])
        by_image[row["image_id"]].add(row["partition"])
    exact = {key: sorted(value) for key, value in by_digest.items() if len(value) > 1}
    groups = {key: sorted(value) for key, value in by_group.items() if len(value) > 1}
    near = {key: sorted(value) for key, value in by_average_hash.items() if len(value) > 1}
    duplicate_images = {key: sorted(value) for key, value in by_image.items() if len(value) > 1}
    return {
        "critical": bool(exact or groups or duplicate_images),
        "exact_duplicate_cross_partition": exact,
        "group_cross_partition": groups,
        "potential_near_duplicate_cross_partition": near,
        "sample_id_cross_partition": duplicate_images,
    }


def validate_manifest(manifest_path: Path, metadata_path: Path, leakage_path: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
        return [f"unable to read split artifacts: {error}"], []
    missing = [field for field in MANIFEST_FIELDS if field not in fields]
    if missing:
        errors.append(f"manifest missing fields: {', '.join(missing)}")
    if not metadata.get("configuration") or "seed" not in metadata["configuration"]:
        errors.append("split metadata does not record configuration and seed")
    seen: defaultdict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        if row["partition"] not in {"train", "validation", "test"}:
            errors.append(f"row {row_number}: invalid partition {row['partition']!r}")
        seen[row["image_id"]].add(row["partition"])
    duplicate_ids = {key: value for key, value in seen.items() if len(value) > 1}
    if duplicate_ids:
        errors.append(f"sample IDs cross partitions: {sorted(duplicate_ids)}")
    if leakage.get("critical"):
        errors.append("critical leakage detected in leakage report")
    if leakage.get("potential_near_duplicate_cross_partition"):
        warnings.append("potential near-duplicate average-hash groups cross partitions")
    return errors, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantvillage-root", type=Path)
    parser.add_argument("--plantdoc-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    entries = load_registry(args.registry, args.metadata)
    all_samples, issues, dataset_stats = [], [], []
    for dataset, root in (("PlantVillage", args.plantvillage_root), ("PlantDoc", args.plantdoc_root)):
        if root is None:
            root = Path("__missing_dataset_root__")
        samples, dataset_issues, stats = scan_dataset(root, dataset, entries)
        all_samples.extend(samples)
        issues.extend({"source_dataset": dataset, **item} for item in dataset_issues)
        dataset_stats.append(stats | {"dataset": dataset})
    splits, warnings = split_samples(all_samples, args.train_ratio, args.validation_ratio, args.seed)
    config = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": 1 - args.train_ratio - args.validation_ratio,
        "unresolved_policy": "exclude",
        "grouping": "sha256 exact-content groups; no invented identifiers",
        "registry": str(args.registry),
    }
    write_outputs(splits, issues, warnings, args.output_dir, args.report_dir, config, dataset_stats)
    print(json.dumps({"eligible_samples": len(all_samples), "excluded_samples": len(issues), "dataset_stats": dataset_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
