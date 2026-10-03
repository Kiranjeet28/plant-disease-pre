import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "plant_disease_efficientnet.keras"
ONNX_PATH = ROOT / "model" / "plant_disease.onnx"
LABELS_PATH = ROOT / "app" / "labels.json"


def load_labels():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if len(labels) != 38:
        raise ValueError(f"Expected 38 labels, found {len(labels)}")
    return labels


def make_test_input(image_path):
    if image_path is None:
        return tf.random.stateless_uniform(
            (1, 224, 224, 3), seed=(2026, 10), dtype=tf.float32
        ).numpy()

    image = tf.keras.utils.load_img(image_path, color_mode="rgb", target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Optional PlantVillage image path")
    args = parser.parse_args()

    labels = load_labels()
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    test_input = make_test_input(args.image)
    keras_output = keras_model(test_input, training=False).numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_input = session.get_inputs()[0]
    onnx_output = session.run(None, {onnx_input.name: test_input})[0]

    keras_index = int(np.argmax(keras_output[0]))
    onnx_index = int(np.argmax(onnx_output[0]))
    maximum_difference = float(np.max(np.abs(keras_output - onnx_output)))

    print("ONNX loaded: YES")
    print(f"Input name: {onnx_input.name}")
    print(f"Input shape: {onnx_input.shape}")
    print(f"Output name: {session.get_outputs()[0].name}")
    print(f"Output shape: {list(onnx_output.shape)}")
    print(f"Number of classes: {onnx_output.shape[-1]}")
    print()
    print(f"Keras output shape: {tuple(keras_output.shape)}")
    print(f"ONNX output shape: {tuple(onnx_output.shape)}")
    print()
    print(f"Keras predicted index: {keras_index}")
    print(f"ONNX predicted index: {onnx_index}")
    print()
    print(f"Keras predicted label: {labels[keras_index]}")
    print(f"ONNX predicted label: {labels[onnx_index]}")
    print()
    print(f"Maximum absolute difference: {maximum_difference:.8g}")

    if keras_output.shape != onnx_output.shape:
        raise AssertionError("Keras and ONNX output shapes differ")
    if keras_index != onnx_index:
        raise AssertionError("Keras and ONNX predicted indices differ")
    if maximum_difference > 1e-4:
        raise AssertionError("Keras and ONNX outputs differ beyond tolerance")

    print()
    print("Keras vs ONNX: PASSED")

    if args.image:
        confidence = float(onnx_output[0, onnx_index]) * 100
        print(f"Predicted index: {onnx_index}")
        print(f"Predicted disease: {labels[onnx_index]}")
        print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()