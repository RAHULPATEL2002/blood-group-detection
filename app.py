from __future__ import annotations

import os
import pickle
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Paths & Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("BLOOD_GROUP_MODEL", "blood_group_model_vgg16.keras")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", "class_indices.pkl")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH = PROJECT_ROOT / "patient_history.db"

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

DEFAULT_CLASS_LABELS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Blood type compatibility data
BLOOD_COMPATIBILITY = {
    "A+":  {"donate_to": ["A+", "AB+"],              "receive_from": ["A+", "A-", "O+", "O-"]},
    "A-":  {"donate_to": ["A+", "A-", "AB+", "AB-"],"receive_from": ["A-", "O-"]},
    "B+":  {"donate_to": ["B+", "AB+"],              "receive_from": ["B+", "B-", "O+", "O-"]},
    "B-":  {"donate_to": ["B+", "B-", "AB+", "AB-"],"receive_from": ["B-", "O-"]},
    "AB+": {"donate_to": ["AB+"],                    "receive_from": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]},
    "AB-": {"donate_to": ["AB+", "AB-"],             "receive_from": ["A-", "B-", "AB-", "O-"]},
    "O+":  {"donate_to": ["A+", "B+", "O+", "AB+"], "receive_from": ["O+", "O-"]},
    "O-":  {"donate_to": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], "receive_from": ["O-"]},
}

BLOOD_FACTS = {
    "A+":  "Second most common blood type. Can donate to A+ and AB+. Found in ~36% of the population.",
    "A-":  "Rare type. Universal plasma donor. Only ~6% of people have A- blood.",
    "B+":  "Found in ~8.5% of population. Especially needed in Asian communities.",
    "B-":  "Very rare — only ~1.5% of people. Important for patients with B- blood.",
    "AB+": "Universal recipient! Can receive any blood type. Rarest common type (~3.4%).",
    "AB-": "Rarest blood type (~0.6%). Universal platelet donor.",
    "O+":  "Most common type (~38%). Can donate red cells to all positive types.",
    "O-":  "Universal donor for red cells. Critical in emergencies. Only ~7% of people.",
}

# ── Model Loading ────────────────────────────────────────────────────────────
MODEL_ERROR: str | None = None
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as exc:
    print(f"❌ Model load error: {exc}")
    MODEL_ERROR = str(exc)
    model = None

class_labels: dict[int, str] = {}
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "rb") as fh:
        idx = pickle.load(fh)
        class_labels = {v: k for k, v in idx.items()}
if not class_labels:
    class_labels = {i: lbl for i, lbl in enumerate(DEFAULT_CLASS_LABELS)}

# ── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT NOT NULL,
            name        TEXT NOT NULL,
            age         INTEGER,
            gender      TEXT,
            blood_type  TEXT NOT NULL,
            confidence  REAL,
            temperature REAL,
            image_path  TEXT,
            notes       TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()

init_db()

def save_record(patient_id, name, age, gender, blood_type, confidence, temperature, image_path, notes=""):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO patient_records (patient_id, name, age, gender, blood_type, confidence, temperature, image_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, name, age, gender, blood_type, confidence, temperature, image_path, notes))
    con.commit()
    record_id = cur.lastrowid
    con.close()
    return record_id

def get_all_records():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM patient_records ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def get_record_by_id(record_id):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM patient_records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None

def delete_record(record_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
    con.commit()
    con.close()

def get_stats():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM patient_records")
    total = cur.fetchone()[0]
    cur.execute("SELECT blood_type, COUNT(*) as cnt FROM patient_records GROUP BY blood_type ORDER BY cnt DESC")
    by_type = [{"blood_type": r[0], "count": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT AVG(confidence)*100 FROM patient_records")
    avg_conf = cur.fetchone()[0] or 0
    con.close()
    return {"total": total, "by_type": by_type, "avg_confidence": round(avg_conf, 1)}

# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def cleanup_uploads(max_age_hours: int = 72) -> None:
    cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
    for fp in UPLOAD_FOLDER.glob("*"):
        try:
            if fp.is_file() and fp.stat().st_mtime < cutoff:
                fp.unlink()
        except OSError:
            continue

def build_confidence_summary(probabilities: np.ndarray) -> list[dict]:
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    return [
        {"label": class_labels.get(i, "Unknown"), "score": float(s), "score_percent": round(float(s) * 100, 1)}
        for i, s in indexed[:4]
    ]

def temperature_note(val: float) -> str:
    if val < 30:
        return "⚠️ Below typical hand surface range. Please recheck sensor."
    if val > 42:
        return "⚠️ High reading for hand surface. Confirm measurement."
    return "✅ Temperature within expected infrared capture range."

@app.context_processor
def inject_globals():
    return {"current_year": datetime.utcnow().year, "model_ready": model is not None}

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    stats = get_stats()
    return render_template("index.html", model_ready=model is not None,
                           model_error=MODEL_ERROR, max_upload_mb=MAX_UPLOAD_MB,
                           default_temperature=36.5, stats=stats)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", model_ready=False, model_error=MODEL_ERROR,
                               error_message="Model not loaded.", max_upload_mb=MAX_UPLOAD_MB,
                               default_temperature=36.5, stats=get_stats()), 500

    # Validate inputs
    name = request.form.get("patient_name", "").strip() or "Anonymous"
    age = request.form.get("patient_age", "").strip()
    gender = request.form.get("patient_gender", "Unknown")
    notes = request.form.get("notes", "").strip()

    try:
        age_val = int(age) if age else None
    except ValueError:
        age_val = None

    if "image" not in request.files or not request.form.get("temperature"):
        return render_template("index.html", model_ready=True,
                               error_message="Image and temperature are required.",
                               max_upload_mb=MAX_UPLOAD_MB, default_temperature=36.5,
                               stats=get_stats()), 400

    img_file = request.files["image"]
    temperature = request.form["temperature"]

    if img_file.filename == "" or not allowed_file(img_file.filename):
        return render_template("index.html", model_ready=True,
                               error_message="Invalid or no file selected.",
                               max_upload_mb=MAX_UPLOAD_MB, default_temperature=36.5,
                               stats=get_stats()), 400

    try:
        temp_val = float(temperature)
    except ValueError:
        return render_template("index.html", model_ready=True,
                               error_message="Temperature must be a number.",
                               max_upload_mb=MAX_UPLOAD_MB, default_temperature=36.5,
                               stats=get_stats()), 400

    cleanup_uploads()
    ext = Path(secure_filename(img_file.filename)).suffix.lower() or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    img_path = UPLOAD_FOLDER / fname
    img_file.save(img_path)

    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array, verbose=0)
        probs = prediction[0]
        class_idx = int(np.argmax(probs))
        blood_type = class_labels.get(class_idx, "Unknown")
        confidence = float(probs[class_idx])

        patient_id = f"PID-{uuid.uuid4().hex[:8].upper()}"
        record_id = save_record(patient_id, name, age_val, gender, blood_type,
                                confidence, temp_val, f"uploads/{fname}", notes)

        compat = BLOOD_COMPATIBILITY.get(blood_type, {})
        fact = BLOOD_FACTS.get(blood_type, "")

        return render_template("result.html", model_ready=True,
                               blood_type=blood_type, temperature=temp_val,
                               temperature_note=temperature_note(temp_val),
                               confidence=confidence,
                               confidence_percent=round(confidence * 100, 1),
                               top_predictions=build_confidence_summary(probs),
                               image_url=f"/static/uploads/{fname}",
                               patient_name=name, patient_age=age_val,
                               patient_gender=gender, patient_id=patient_id,
                               record_id=record_id, notes=notes,
                               donate_to=compat.get("donate_to", []),
                               receive_from=compat.get("receive_from", []),
                               blood_fact=fact,
                               timestamp=datetime.utcnow().strftime("%B %d, %Y  %H:%M UTC"))
    except Exception as exc:
        return render_template("index.html", model_ready=True,
                               error_message=f"Processing error: {exc}",
                               max_upload_mb=MAX_UPLOAD_MB, default_temperature=36.5,
                               stats=get_stats()), 500

@app.route("/history")
def history():
    records = get_all_records()
    stats = get_stats()
    return render_template("history.html", records=records, stats=stats, model_ready=model is not None)

@app.route("/history/delete/<int:record_id>", methods=["POST"])
def delete_history(record_id):
    delete_record(record_id)
    return jsonify({"success": True})

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}, 200

@app.errorhandler(413)
def file_too_large(error):
    return render_template("index.html", model_ready=model is not None,
                           error_message=f"File too large. Max {MAX_UPLOAD_MB} MB.",
                           max_upload_mb=MAX_UPLOAD_MB, default_temperature=36.5,
                           stats=get_stats()), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
