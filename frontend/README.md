# Plant Disease Detection frontend

This React/Vite interface sends an explicitly selected JPG, JPEG, or PNG image
to the backend Step 9 router at `POST /predict/router`. The frontend validates
the file type and 10 MB size limit for usability; the backend remains
authoritative and does not permanently store uploaded images.

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

The interface displays the router's canonical display label, healthy status,
confidence only when a prediction is available, calibration status, selected
model, inference duration, warnings, and abstention reason. Uncalibrated
scores are explicitly labelled as model scores rather than probabilities.
`uncertain` and `unavailable` router responses are never rendered as disease
predictions. Network and server errors expose a retryable message without
showing stack traces or filesystem paths.

## Checks

```powershell
npm run lint
npm run build
```

The current backend router has one verified ready model and may return an
unavailable result for unsupported crops or missing/incompatible artifacts.
