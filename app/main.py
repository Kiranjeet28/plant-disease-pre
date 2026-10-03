import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "plant_disease.onnx"
LABELS_PATH = ROOT / "app" / "labels.json"

labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
if len(labels) != 38:
    raise RuntimeError(f"Expected 38 labels, found {len(labels)}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

app = FastAPI(title="Plant Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://plant-disease-pre.onrender.com",
        "https://plant-disease-pre.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_array = np.asarray(
                image.convert("RGB").resize((224, 224), Image.Resampling.NEAREST),
                dtype=np.float32,
            )
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from error

    return np.expand_dims(image_array, axis=0)


def predict(image_bytes: bytes) -> tuple[int, float]:
    image_input = prepare_image(image_bytes)
    output = np.asarray(session.run(None, {input_name: image_input})[0][0])
    if output.shape != (len(labels),):
        raise RuntimeError(f"Unexpected model output shape: {output.shape}")

    class_index = int(np.argmax(output))
    confidence = float(output[class_index])
    return class_index, confidence


@app.get("/")
def root():
    return {"status": "online", "model": "Plant Disease EfficientNetB0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing image file")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        class_index, confidence = predict(image_bytes)
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail="Unable to process image") from error

    return {
        "disease": labels[class_index],
        "confidence": round(confidence * 100, 2),
        "class_index": class_index,
    }