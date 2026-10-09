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
