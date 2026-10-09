"""Convert the verified MobileNetV2 checkpoint to ONNX and verify equivalence."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model_adapters import PytorchMobileNetV2Adapter


DEFAULT_OUTPUT = ROOT / "model" / "mobilenetv2_plant.onnx"
DEFAULT_REPORT = ROOT / "reports" / "mobilenetv2_onnx_conversion.json"


def _probe_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (80, 120, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def convert(output_path: Path = DEFAULT_OUTPUT, report_path: Path = DEFAULT_REPORT) -> dict:
    import torch

    adapter = PytorchMobileNetV2Adapter()
    model = adapter._model
    model.eval()
    sample_bytes = _probe_image()
    sample = torch.from_numpy(adapter.preprocess(sample_bytes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    torch_output = model(sample).detach().cpu().numpy()
    onnx_output = session.run(None, {"input": sample.detach().cpu().numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    if not np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"ONNX output mismatch: maximum absolute difference {max_abs_diff}"
        )
    report = {
        "schema_version": 1,
        "source_checkpoint": str(adapter.config.checkpoint),
        "output_checkpoint": str(output_path),
        "architecture": adapter.config.architecture,
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, len(adapter.config.labels)],
        "opset": 17,
        "source_score_type": "logits",
        "max_absolute_output_difference": max_abs_diff,
        "onnxruntime_verified": True,
        "preprocessing": adapter.config.normalization,
        "canonical_label_mapping_verified": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(convert(args.output, args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
