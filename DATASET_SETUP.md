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
