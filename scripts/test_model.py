import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "plant_disease_efficientnet.keras"
LABELS_PATH = ROOT / "app" / "labels.json"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    if len(labels) != 38:
        raise ValueError(f"Expected 38 labels, found {len(labels)}")

    input_shape = tuple(model.input_shape)
    if input_shape != (None, 224, 224, 3):
        raise ValueError(f"Unexpected model input shape: {input_shape}")

    dummy_input = tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)
    output = model(dummy_input, training=False).numpy()
    output_shape = tuple(output.shape)
    if output_shape != (1, len(labels)):
        raise ValueError(
            f"Model output shape {output_shape} does not match {len(labels)} labels"
        )

    predicted_index = int(np.argmax(output[0]))
    print("Model loaded: YES")
    print(f"Classes: {len(labels)}")
    print("Labels loaded: YES")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted label: {labels[predicted_index]}")


if __name__ == "__main__":
    main()