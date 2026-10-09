# Plant Disease Detection frontend

This React/Vite interface sends an explicitly selected JPG, JPEG, or PNG image
to the backend analysis endpoint at `POST /analyze`. The frontend validates the
file type and 10 MB size limit for usability; the backend remains authoritative
and does not permanently store uploaded images.

## Run locally

From `frontend`:

```powershell
npm run dev
```

Set `VITE_API_URL` in `.env` when the API is not running at
`http://127.0.0.1:8000`. Start the backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Use **Choose plant image** or drag a file into the upload area. Prediction
starts only after pressing **Predict disease**. Selecting a replacement image
clears the previous result, and an in-flight request cannot be submitted twice.

## Results and safety states

The dashboard displays actual evaluated model cards, excluded model reasons,
top predictions, score types, latency, canonical display labels, healthy
status, calibration status, warnings, and abstention reasons. The score chart
contains current-image scores only. Offline benchmark metrics are shown in a
separate section and remain `Not available` while the verified test manifest
has no samples. Uncalibrated scores are explicitly labelled as model scores
rather than probabilities. `uncertain` and `unavailable` responses are never
rendered as disease predictions.

The decision trace reflects the real router: it does not claim voting,
averaging, or consensus. The capability registry contains verified ONNX,
Keras, and MobileNetV2 adapters; ONNX remains the highest-priority router
choice.

## Checks

```powershell
npm run lint
npm run build
```

The current backend router has two verified PlantVillage EfficientNet adapters
(ONNX and Keras), with ONNX configured as the higher-priority production
model. The MobileNet checkpoint uses its verified MobileNetV2 architecture and
ImageNet preprocessing. It can still return an unavailable result when the
PyTorch runtime is missing or the checkpoint fails strict validation.
