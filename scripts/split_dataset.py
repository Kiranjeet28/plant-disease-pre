"""Audit and reproducibly split the existing 38-class image dataset.

The source dataset is never modified. This utility copies files into a new
output directory after validating labels, images, duplicates, and splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Iterable

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "dataset_config.json"
DEFAULT_LABELS = ROOT / "app" / "labels.json"
DEFAULT_REPORT = ROOT / "reports" / "dataset_split_report.json"
DEFAULT_OUTPUT = ROOT / "data_split"


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    group: str
    digest: str
    width: int
    height: int
    channels: int


def load_labels(path: Path = DEFAULT_LABELS) -> list[str]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) != 38 or len(set(labels)) != 38:
        raise ValueError("The labels file must contain 38 unique labels.")
    return labels


def _source_group(path: Path, digest: str) -> str:
    """Group exact duplicates and common PlantVillage derivative filenames."""
    # Hash grouping catches byte-identical files regardless of filename.
    # The normalized name groups common augmented copies (for example
    # leaf_flip, leaf_rot90, and leaf_augmented_1) without changing labels.
    stem = path.stem.casefold()
    normalized = re.sub(
        r"([_-](?:aug(?:mented)?|flip|flipped|rot(?:ation)?|crop|zoom)"
        r"[_-]?\d*)+$",
        "",
        stem,
    )
    return f"source:{normalized}" if normalized != stem else f"hash:{digest}"


def _image_files(folder: Path, extensions: set[str]) -> Iterable[Path]:
    return (
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    )


def inspect_dataset(
    dataset_root: Path, labels: list[str], extensions: set[str]
) -> tuple[dict, list[Record]]:
    report = {
        "dataset_root": str(dataset_root),
        "status": "ok",
        "missing_classes": [],
        "extra_classes": [],
        "empty_classes": [],
        "unsupported_files": [],
        "corrupt_images": [],
        "counts": {label: 0 for label in labels},
        "dimensions": Counter(),
        "channels": Counter(),
        "duplicate_groups": {},
        "records": 0,
    }
    if not dataset_root.exists():
        report["status"] = "missing_dataset"
        report["error"] = (
            "Dataset root does not exist. Supply --dataset-root pointing to the "
            "original PlantVillage directory."
        )
        return report, []
    if not dataset_root.is_dir():
        report["status"] = "invalid_dataset_root"
        report["error"] = "Dataset root is not a directory."
        return report, []

    class_dirs = {path.name: path for path in dataset_root.iterdir() if path.is_dir()}
    report["missing_classes"] = [label for label in labels if label not in class_dirs]
    report["extra_classes"] = sorted(set(class_dirs) - set(labels))
    records: list[Record] = []
    digest_paths: defaultdict[str, list[str]] = defaultdict(list)

    for label in labels:
        folder = class_dirs.get(label)
        if folder is None:
            continue
        image_paths = list(_image_files(folder, extensions))
        if not image_paths:
            report["empty_classes"].append(label)
        known_files = {
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() not in extensions
        }
        report["unsupported_files"].extend(str(path) for path in sorted(known_files))
        for path in image_paths:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    width, height = image.size
                    channels = len(image.getbands())
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                report["corrupt_images"].append({"path": str(path), "error": str(error)})
                continue
            record = Record(
                path=path,
                label=label,
                group=_source_group(path, digest),
                digest=digest,
                width=width,
                height=height,
                channels=channels,
            )
            records.append(record)
            report["counts"][label] += 1
            report["dimensions"][f"{width}x{height}"] += 1
            report["channels"][str(channels)] += 1
            digest_paths[digest].append(str(path))

    digest_counts = Counter(record.digest for record in records)
    records = [
        replace(record, group=f"hash:{record.digest}")
        if digest_counts[record.digest] > 1
        else record
        for record in records
    ]
    report["records"] = len(records)
    report["dimensions"] = dict(sorted(report["dimensions"].items()))
    report["channels"] = dict(sorted(report["channels"].items()))
    report["duplicate_groups"] = {
        digest: paths for digest, paths in sorted(digest_paths.items()) if len(paths) > 1
    }
    if report["missing_classes"] or report["extra_classes"] or report["empty_classes"]:
        report["status"] = "incompatible_classes"
    if report["corrupt_images"]:
        report["status"] = "invalid_images"
    return report, records


def _allocate_groups(records: list[Record], ratio: float, rng: Random) -> set[str]:
    groups = sorted({record.group for record in records})
    rng.shuffle(groups)
    count = round(len(groups) * ratio)
    return set(groups[:count])


def split_records(
    records: list[Record], train_ratio: float, validation_ratio: float, seed: int
) -> dict[str, list[Record]]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must be positive and leave test data.")
    rng = Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    grouped: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    group_items = list(grouped.values())
    rng.shuffle(group_items)
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": 1 - train_ratio - validation_ratio}
    label_totals = Counter(record.label for record in records)
    label_targets = {
        split: {label: total * ratio for label, total in label_totals.items()}
        for split, ratio in ratios.items()
    }
    for group_records in group_items:
        # Pick the split furthest below its per-class target. The whole group
        # is assigned atomically, including duplicates across class folders.
        counts = Counter(record.label for record in group_records)
        split = max(
            splits,
            key=lambda candidate: sum(
                label_targets[candidate][label]
                - sum(item.label == label for item in splits[candidate])
                for label in counts
            ),
        )
        splits[split].extend(group_records)
    return splits


def write_split(
    splits: dict[str, list[Record]], output_root: Path, labels: list[str]
) -> None:
    for split, records in splits.items():
        for record in records:
            destination = output_root / split / record.label / record.path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.path, destination)
    (output_root / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = load_labels(ROOT / config["labels_file"])
    extensions = {extension.lower() for extension in config["image_extensions"]}
    report, records = inspect_dataset(args.dataset_root, labels, extensions)
    report.update(
        {
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": 1 - args.train_ratio - args.validation_ratio,
            },
            "split_counts": {},
            "split_class_counts": {},
            "representation_exceptions": [],
            "overlapping_files_between_splits": {},
        }
    )
    if report["status"] == "ok" and records:
        splits = split_records(records, args.train_ratio, args.validation_ratio, args.seed)
        seen: dict[str, str] = {}
        split_paths: dict[str, set[str]] = {}
        for split, split_records_list in splits.items():
            split_paths[split] = {str(record.path.resolve()) for record in split_records_list}
            report["split_counts"][split] = len(split_records_list)
            report["split_class_counts"][split] = dict(
                Counter(record.label for record in split_records_list)
            )
            for record in split_records_list:
                if record.group in seen and seen[record.group] != split:
                    raise AssertionError("A source group was assigned to multiple splits.")
                seen[record.group] = split
        report["overlapping_files_between_splits"] = {
            f"{left}:{right}": sorted(split_paths[left] & split_paths[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
            if split_paths[left] & split_paths[right]
        }
        if report["overlapping_files_between_splits"]:
            raise AssertionError("The same source file was assigned to multiple splits.")
        for label in labels:
            missing_splits = [
                split
                for split in splits
                if not any(record.label == label for record in splits[split])
            ]
            if missing_splits:
                report["representation_exceptions"].append(
                    {"class": label, "missing_splits": missing_splits}
                )
        if report["representation_exceptions"]:
            report["status"] = "insufficient_class_representation"
        if args.output_root:
            if report["status"] == "ok":
                write_split(splits, args.output_root, labels)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "ok" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--output-root", type=Path, default=None)
    result.add_argument("--seed", type=int, default=20261009)
    result.add_argument("--train-ratio", type=float, default=0.70)
    result.add_argument("--validation-ratio", type=float, default=0.15)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
