"""Validate registered model artifacts and optionally run a real image probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference_router import InferenceRouter, load_capabilities
from scripts.model_adapters import AdapterError


def diagnose(image_path: Path | None = None) -> dict[str, Any]:
    router = InferenceRouter()
    image_bytes = image_path.read_bytes() if image_path else None
    models = []
    for capability in router.capabilities:
        record: dict[str, Any] = {
            "model_id": capability.model_id,
            "adapter_type": capability.adapter_type,
            "configured_status": capability.status,
            "supported_crops": list(capability.supported_crops),
            "checkpoint_available": None,
            "adapter_initialized": False,
            "metadata": None,
            "inference": None,
            "readiness": capability.status,
            "reasons": list(capability.limitations),
        }
        if capability.status == "unavailable":
            record["reasons"].append("Configured unavailable; adapter initialization was not attempted.")
            models.append(record)
            continue
        try:
            adapter = router._get_adapter(capability.model_id)
            record["adapter_initialized"] = True
            record["checkpoint_available"] = adapter.config.checkpoint.is_file()
            record["metadata"] = adapter.metadata()
            if image_bytes is not None:
                prediction = adapter.predict(image_bytes)
                record["inference"] = prediction.to_dict()
                record["readiness"] = "ready"
        except (AdapterError, OSError, ValueError, RuntimeError) as error:
            record["readiness"] = (
                "unavailable" if capability.status == "unavailable"
                else "failed_validation"
            )
            record["reasons"].append(str(error))
        models.append(record)
    return {
        "schema_version": 1,
        "image_probe": str(image_path) if image_path else None,
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional image for real inference probes.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "model_diagnostics.json")
    args = parser.parse_args()
    if args.image and not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    report = diagnose(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(model["readiness"] in {"ready", "requires_review", "unavailable"}
                    for model in report["models"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
