# Dataset setup and split audit

The repository does not contain the PlantVillage images. Place the original,
unmodified dataset outside Git, then pass its absolute path to the utility.
The expected layout uses the exact names in `app/labels.json`:

```text
<dataset-root>\
  Apple___Apple_scab\
    image files
  ...
  Tomato___healthy\
    image files
```

The utility does not download, alter, rename, or augment source images. It
rejects missing/extra/empty classes and corrupt images, reports unsupported
files, dimensions, channels, exact duplicate groups, and writes the preserved
label order to the output split.

## Windows PowerShell

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\split_dataset.py `
  --dataset-root 'D:\datasets\PlantVillage' `
  --report .\reports\dataset_split_report.json `
  --output-root .\data_split `
  --seed 20261009 `
  --train-ratio 0.70 `
  --validation-ratio 0.15
```

The command returns exit code `2` until a compatible dataset is supplied.
Validation and test directories contain copied, unaugmented source images;
augmentation belongs only in the later training input pipeline.

Run the utility tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Dataset audit (Step 2)

The audit is read-only with respect to source datasets. It does not download
PlantVillage or PlantDoc, create splits, or remap labels. Supply local roots
only when the original datasets have been downloaded outside Git:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_datasets.py `
  --plantvillage-root 'D:\datasets\PlantVillage' `
  --plantdoc-root 'D:\datasets\PlantDoc' `
  --output-dir .\reports\dataset_audit
```

If a root is omitted or does not exist, the report records that dataset as
`not_available`; it does not fabricate local counts. Reports include JSON and
CSV summaries, image integrity issues, SHA-256 duplicate groups, and
conservative average-hash candidates that require manual review. PlantDoc
classification folders and object-detection annotations are reported
separately and must not be treated as interchangeable.

## Canonical disease registry (Step 3)

The versioned registry is stored in
`data/label_registry.csv`, with schema and provenance in
`data/label_registry_metadata.json`. Validate it without loading the model:

```powershell
.\.venv\Scripts\python.exe .\scripts\label_registry.py
```

The command exits with status `0` when the registry is valid and status `2`
when errors prevent safe use. Lookup helpers are available from
`scripts.label_registry`. The current registry contains only the 38 exact,
verified PlantVillage labels already present in `app/labels.json`. PlantDoc
has no local class labels in this checkout, so it remains documented as
unresolved and is not fabricated into the registry.

## End-to-end integration checks (Step 11)

The integration suite exercises the upload endpoint, router response contract,
canonical registry lookups, healthy and diseased result states, abstention and
image-quality warnings, unsupported crops, upload limits, and model/service
failure responses. It uses a small generated image fixture and controlled
adapter responses; it does not require local datasets or model downloads.

Run it with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_end_to_end_integration -v
```

Run all backend regression and integration tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The real-model integration case verifies that any returned canonical ID,
display label, and healthy status agree with the canonical registry. A
synthetic solid-colour image is only an execution fixture; its model output
must not be interpreted as a validated plant disease or healthy-leaf example.

## Model analysis dashboard

The frontend analysis dashboard uses `POST /analyze` rather than simulating
multiple models in the browser. The response includes the actual evaluated
adapter output, score type, top source-label scores, canonical mapping,
latency, warnings, and explicit exclusion reasons for models that are not
ready or do not support the requested crop. The current capability registry
has one ready ONNX model; Keras and PyTorch checkpoints are not presented as
evaluated predictions.

The current-image score comparison is separate from offline benchmark
performance. Because the verified test manifest is empty, benchmark values
are displayed as unavailable rather than fabricated. Start the backend and
frontend as documented in `frontend/README.md`, then upload an image and select
**Analyze image**.

Run the repeatable model readiness diagnostic with an optional image probe:

```powershell
.\.venv\Scripts\python.exe .\scripts\diagnose_models.py `
  --image 'D:\images\leaf.jpg' `
  --output .\reports\model_diagnostics.json
```

The current diagnostic verifies real inference for the ONNX and Keras
EfficientNet checkpoints. The MobileNet checkpoint remains unavailable because
the original PyTorch source is retained separately while the converted
`model/mobilenetv2_plant.onnx` artifact is used for fast inference. The
conversion report records PyTorch/ONNX output equivalence and the diagnostic
does not modify the original `.pth` checkpoint.

Convert or regenerate the MobileNet ONNX artifact with:

```powershell
.\.venv\Scripts\python.exe .\scripts\convert_mobilenet_to_onnx.py
```

The converter validates the ONNX graph and compares its logits with the
PyTorch model on the same preprocessed input before writing the report.

## Safe dataset splits (Step 4)

Create reference-only manifests without copying or modifying source images:

```powershell
.\.venv\Scripts\python.exe .\scripts\create_dataset_splits.py `
  --plantvillage-root 'D:\datasets\PlantVillage' `
  --plantdoc-root 'D:\datasets\PlantDoc' `
  --output-dir .\data\manifests `
  --report-dir .\reports `
  --seed 20261009 `
  --train-ratio 0.70 `
  --validation-ratio 0.15
```

Validate generated artifacts:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from scripts.create_dataset_splits import validate_manifest; import sys; e,w=validate_manifest(Path('data/manifests/split_manifest.csv'), Path('data/manifests/split_metadata.json'), Path('reports/data_leakage_report.json')); print({'errors':e,'warnings':w}); sys.exit(2 if e else 0)"
```

The splitter excludes unresolved labels and corrupt images into
`data/manifests/excluded_samples.csv`, keeps SHA-256 duplicate groups in one
partition, reports perceptual-hash candidates, and records unavailable dataset
roots explicitly. It does not replace official dataset splits or create model
training data.

## Candidate model training (Step 5)

Training consumes the verified Step 4 manifest and never creates a new split.
It is intentionally blocked when the manifest has no eligible samples:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_models.py `
  --manifest .\data\manifests\split_manifest.csv `
  --split-metadata .\data\manifests\split_metadata.json `
  --architecture small_cnn `
  --epochs 10 `
  --batch-size 32 `
  --output-dir .\experiments
```

Available architectures are `small_cnn` and `efficientnet_b0`. The latter can
request verified Keras ImageNet weights with `--pretrained`; this may download
weights and should not be used until the dataset prerequisite is available.
The current checkout has zero eligible samples, so no training run or metrics
were produced. The existing ONNX production model remains untouched.

## Benchmarking and evaluation (Step 6)

Benchmarking evaluates only the verified Step 4 test partition and does not
train, calibrate, or modify models:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_models.py `
  --manifest .\data\manifests\split_manifest.csv `
  --report-dir .\reports
```

The command writes `benchmark_summary.json`, `model_comparison.csv`,
`per_class_metrics.csv`, and `latency_benchmark.json`. With the current empty
test manifest it records the baseline and candidates as skipped rather than
inventing metrics. Metric helpers support accuracy, macro/weighted F1,
per-class precision/recall/F1, confusion matrices, top-k accuracy, and
repeatable warm-up-excluded latency measurements.

## Model adapters (Step 8)

The adapter interface is implemented in
`scripts/model_adapters.py`. The verified ready adapter is
`plantvillage-efficientnetb0-onnx`, backed by
`model/plant_disease.onnx`, with 224×224 RGB input, raw float32 pixel values,
38 probability outputs, and the exact source labels from `app/labels.json`.
Each source label is checked against the verified PlantVillage canonical
registry before inference.

The Keras checkpoint is marked `ready` but remains uncalibrated because no
compatible validation-derived calibration artifact is available. The MobileNet
PyTorch checkpoint is verified against its published model documentation:
MobileNetV2, 224×224 RGB input, ImageNet normalization, a 38-class logits head,
and the exact PlantVillage class order. Its adapter applies softmax once for
display probabilities and reports the result as uncalibrated.

Adapters return raw scores separately from calibrated probabilities,
canonical mapping, calibration status, abstention status, image-quality
warnings, and inference duration. Step 7 calibration is not applied to the
ONNX probability output because the available calibration method is
logit-based and no compatible artifact exists. No inference router is added in
Step 8.

Run adapter tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_adapters -v
```

## Inference router (Step 9)

The routing layer is implemented in
`scripts/inference_router.py` and configured by
`data/model_capabilities.json`. It is intentionally separate from
`app/main.py`; the existing public `/predict` endpoint is unchanged.

Routing flow:

1. Reject empty input.
2. Select only capability entries marked `ready`.
3. Filter by an explicitly supplied crop when available.
4. Use stable `model_id` ordering as the documented tie-breaker because
   benchmark metrics are currently unavailable.
5. Reuse a cached Step 8 adapter.
6. Propagate raw scores, calibration status, abstention, image-quality warnings,
   canonical mapping, and inference duration.
7. Return an explicit unavailable or uncertain response on failure.

The current registry marks the ONNX PlantVillage adapter as ready, the Keras
checkpoint as ready but uncalibrated, and the MobileNet checkpoint as ready
when its PyTorch runtime and strict checkpoint validation succeed.
No models are combined, and no unsupported crop or class is silently mapped.

Run router tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_inference_router -v
```

## Confidence calibration and uncertainty (Step 7)

Calibration is implemented separately from production inference and uses
validation predictions only. It supports temperature scaling for verified
logits, NLL, Brier score, ECE, reliability data, validation-derived
abstention thresholds, and conservative image-quality checks:

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_models.py `
  --manifest .\data\manifests\split_manifest.csv `
  --output-dir .\reports
```

With the current empty validation manifest, the command writes explicit
`unavailable` calibration, abstention, OOD, and reliability reports. It does
not infer predictions, fit parameters, modify model weights, change the
prediction API, or claim OOD performance. Calibration artifacts are stored
separately and include the model class ordering when fitting is later enabled.
