from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import pickle
import uuid

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("BLOOD_GROUP_MODEL", "blood_group_model_vgg16.keras")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", "class_indices.pkl")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

DEFAULT_CLASS_LABELS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
MODEL_ERROR: str | None = None

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as exc:
    print(f"❌ Error loading model: {exc}")
    MODEL_ERROR = str(exc)
    model = None

class_labels: dict[int, str] = {}
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "rb") as handle:
        class_indices = pickle.load(handle)
    class_labels = {v: k for k, v in class_indices.items()}

if not class_labels:
    class_labels = {index: label for index, label in enumerate(DEFAULT_CLASS_LABELS)}


@app.context_processor
def inject_globals():
    return {
        "current_year": datetime.utcnow().year,
        "model_ready": model is not None,
    }


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def cleanup_uploads(max_age_hours: int = 24) -> None:
    cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
    for file_path in UPLOAD_FOLDER.glob("*"):
        try:
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                file_path.unlink()
        except OSError:
            continue


def build_confidence_summary(probabilities: np.ndarray) -> list[dict[str, object]]:
    indexed = list(enumerate(probabilities))
    indexed.sort(key=lambda item: item[1], reverse=True)
    summary: list[dict[str, object]] = []
    for index, score in indexed[:3]:
        summary.append(
            {
                "label": class_labels.get(index, "Unknown"),
                "score": float(score),
                "score_percent": round(float(score) * 100, 1),
            }
        )
    return summary


def temperature_note(value: float) -> str:
    if value < 30:
        return "Temperature is below typical hand surface range. Recheck the sensor reading."
    if value > 42:
        return "Temperature looks high for hand surface readings. Confirm the measurement."
    return "Temperature is within the expected surface range for infrared capture."


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_ready=model is not None,
        model_error=MODEL_ERROR,
        max_upload_mb=MAX_UPLOAD_MB,
        default_temperature=36.5,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            render_template(
                "index.html",
                model_ready=False,
                model_error=MODEL_ERROR,
                error_message="Model is not loaded on the server. Please deploy the model file first.",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            500,
        )

    if "image" not in request.files or "temperature" not in request.form:
        return (
            render_template(
                "index.html",
                model_ready=True,
                error_message="Image and temperature input are required.",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            400,
        )

    img_file = request.files["image"]
    temperature = request.form["temperature"]

    if img_file.filename == "" or not temperature:
        return (
            render_template(
                "index.html",
                model_ready=True,
                error_message="No file selected or temperature missing.",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            400,
        )

    if not allowed_file(img_file.filename):
        return (
            render_template(
                "index.html",
                model_ready=True,
                error_message="Unsupported file format. Please upload a JPG, PNG, WEBP, or BMP image.",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            400,
        )

    try:
        temp_value = float(temperature)
    except ValueError:
        return (
            render_template(
                "index.html",
                model_ready=True,
                error_message="Temperature must be a valid number.",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            400,
        )

    cleanup_uploads()
    extension = Path(secure_filename(img_file.filename)).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension or '.jpg'}"
    img_path = UPLOAD_FOLDER / unique_name
    img_file.save(img_path)

    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        probabilities = prediction[0]
        class_index = int(np.argmax(probabilities))
        blood_type = class_labels.get(class_index, "Unknown")
        confidence = float(probabilities[class_index])

        return render_template(
            "result.html",
            model_ready=True,
            blood_type=blood_type,
            temperature=temp_value,
            temperature_note=temperature_note(temp_value),
            confidence=confidence,
            confidence_percent=round(confidence * 100, 1),
            top_predictions=build_confidence_summary(probabilities),
            image_url=f"/static/uploads/{unique_name}",
        )
    except Exception as exc:
        return (
            render_template(
                "index.html",
                model_ready=True,
                error_message=f"Error processing image: {exc}",
                max_upload_mb=MAX_UPLOAD_MB,
                default_temperature=36.5,
            ),
            500,
        )


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}, 200


@app.errorhandler(413)
def file_too_large(error):
    return (
        render_template(
            "index.html",
            model_ready=model is not None,
            error_message=f"File is too large. Max upload size is {MAX_UPLOAD_MB} MB.",
            max_upload_mb=MAX_UPLOAD_MB,
            default_temperature=36.5,
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
