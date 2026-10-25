import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path="", static_folder=".")

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = [
    "Coronary Artery Disease",
    "Cardiomyopathy",
    "Heart Failure",
    "Myocardial Infarction",
    "Arrhythmia",
    "Valvular Heart Disease",
    "Congenital Heart Disease",
    "Normal / No Abnormality",
]
MODEL_INPUT_SIZE = (224, 224)
MODEL_SEARCH_PATHS = [
    "models/heart_cnn.keras",
    "models/heart_cnn.h5",
    "heart_cnn.keras",
    "heart_cnn.h5",
    "trained_model.keras",
    "trained_model.h5",
]


def build_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_saved_model(model_path):
    if not os.path.exists(model_path):
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        if model is not None:
            return model
    except Exception:
        return None

    return None


def initialize_model():
    for path in MODEL_SEARCH_PATHS:
        model = load_saved_model(path)
        if model is not None:
            return model

    model = build_cnn_model()
    dummy_input = np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), dtype=np.float32)
    model(dummy_input)
    return model


MODEL = initialize_model()
MODEL_SOURCE = "trained" if any(os.path.exists(path) for path in MODEL_SEARCH_PATHS) else "starter"


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(".", "style.css")


@app.route("/script.js")
def serve_js():
    return send_from_directory(".", "script.js")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_source": MODEL_SOURCE})


def preprocess_image(raw_image_bytes):
    image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
    image = image.resize(MODEL_INPUT_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array[np.newaxis, ...]


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Please select an image file first."}), 400

    filename = secure_filename(uploaded_file.filename)
    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Please upload a JPG, JPEG, PNG, or WEBP image."}), 400

    file_bytes = uploaded_file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "Image must be smaller than 10 MB."}), 400

    try:
        image_tensor = preprocess_image(file_bytes)
        probabilities = MODEL.predict(image_tensor, verbose=0)[0]
    except Exception as exc:
        return jsonify({"error": f"Image processing failed: {str(exc)}"}), 400

    probability_list = [
        {"name": class_name, "score": int(round(float(probability) * 100))}
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    ]

    best_result = max(probability_list, key=lambda item: item["score"])

    return jsonify(
        {
            "predicted_condition": best_result["name"],
            "confidence": best_result["score"],
            "probabilities": probability_list,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
