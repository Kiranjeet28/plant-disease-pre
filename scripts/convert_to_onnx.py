from pathlib import Path

import tensorflow as tf
import tf2onnx


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "plant_disease_efficientnet.keras"
ONNX_PATH = ROOT / "model" / "plant_disease.onnx"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    # tf2onnx 1.16 expects this legacy Keras attribute when converting Keras 3 models.
    if not hasattr(model, "output_names"):
        model.output_names = ["output"]
    input_signature = (
        tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
    )

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=17,
        output_path=str(ONNX_PATH),
    )

    if not ONNX_PATH.exists() or ONNX_PATH.stat().st_size == 0:
        raise RuntimeError("ONNX export did not produce a valid file")

    print("ONNX conversion successful")
    print(f"Model path: {ONNX_PATH.relative_to(ROOT)}")
    print(f"Size: {ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()