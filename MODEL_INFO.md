# Plant Disease EfficientNet Model

## Source

- Hugging Face model: [Nefflymicn/PlantVillage-plant-disease-detection](https://huggingface.co/Nefflymicn/PlantVillage-plant-disease-detection)
- Original training project: [JinWei225/Plant-disease-detection](https://github.com/JinWei225/Plant-disease-detection)
- License: Apache-2.0.
- Model file: `plant_disease_efficientnet.keras` (38,655,526 bytes).
- Label file: `class_names.txt`, copied to [app/labels.json](app/labels.json) without reordering.

## Verified model details

- Framework: TensorFlow/Keras.
- Architecture: EfficientNetB0 with GlobalAveragePooling2D, BatchNormalization, Dense(512, ReLU), Dropout(0.45), Dense(256, ReLU), Dropout(0.35), and Dense(38, Softmax).
- Number of classes: 38.
- Input image size: `224 x 224` RGB.
- Model input shape: `(None, 224, 224, 3)`.
- Preprocessing: TensorFlow Keras EfficientNet `preprocess_input`, as used by the original inference code and training project. EfficientNet preprocessing is built into the Keras EfficientNet model for this input path; no separate ImageNet mean/std transform is applied.
- Label ordering: the repository's `class_names.txt` explicitly assigns indices `0` through `37`; the downloaded list contains exactly 38 entries and matches the model's documented 38-output head.

## Exact label ordering

The order below is copied directly from the candidate repository's indexed `class_names.txt` and is also stored in `app/labels.json`:

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

The source README and original GitHub training project both describe the classifier as a 38-class EfficientNetB0 model and identify `class_names.txt` as the class-index mapping. The local test script verifies that the loaded model output width equals the 38 labels.

## ONNX conversion

- ONNX conversion: successful.
- Conversion method: `tf2onnx.convert.from_keras`.
- ONNX runtime verification: successful.
- Keras vs ONNX prediction: matched.
- ONNX input shape: `[batch, 224, 224, 3]`.
- ONNX output shape: `[batch, 38]`.
- ONNX opset: 17.
- ONNX file: `model/plant_disease.onnx`.
- ONNX file size: 19,279,389 bytes (18.39 MB).
- Same-input comparison: Keras and ONNX both predicted index 30 (`Tomato___Late_blight`); maximum absolute output difference was `5.2154064e-08`.
