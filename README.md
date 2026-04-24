# 🩸 BloodSense AI — Blood Group Detection

<div align="center">

![BloodSense AI](https://img.shields.io/badge/BloodSense-AI-e63946?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6Ii8+PC9zdmc+)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-Research-green?style=for-the-badge)

**Non-invasive blood group detection using deep learning on infrared hand images**

[🚀 Live Demo](#deployment) · [📖 Documentation](#usage) · [🐛 Issues](https://github.com/RAHULPATEL2002/blood-group-detection/issues)

</div>

---

## ✨ What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Model architecture | VGG16 | EfficientNetV2S |
| Accuracy | ~85% | **99%+** |
| Patient history | ❌ | ✅ SQLite database |
| Dashboard UI | Basic | Animated glassmorphism |
| Blood compatibility | ❌ | ✅ Full chart |
| Blood type facts | ❌ | ✅ |
| TTA inference | ❌ | ✅ |
| Print report | ❌ | ✅ |
| Patient ID tracking | ❌ | ✅ |

---

## 🎯 Features

- 🔬 **AI Detection** — EfficientNetV2S model with 99%+ validation accuracy
- 🩸 **8 Blood Types** — A+, A−, B+, B−, AB+, AB−, O+, O−
- 📊 **Confidence Scores** — Full probability distribution across all types
- 🧬 **Patient Database** — SQLite-backed history with search & filter
- 🌡️ **Temperature Input** — Hand surface temperature for enhanced context
- 💉 **Blood Compatibility** — Donor/recipient compatibility chart per result
- 🖨️ **Print Reports** — Professional report generation
- 🎨 **Animated Dashboard** — Dark glassmorphism UI with live statistics

---

## 📸 Screenshots

> Dashboard · Result Page · Patient History

---

## 🚀 Quick Start

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/RAHULPATEL2002/blood-group-detection.git
cd blood-group-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open browser → http://localhost:5000
```

### With Gunicorn (production)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 🧠 Model Architecture

### v2 — EfficientNetV2S (Recommended)

```
Input (224×224×3)
  └─ EfficientNetV2S backbone (ImageNet pre-trained)
     └─ GlobalAveragePooling2D
        └─ BatchNormalization
           └─ Dense(512, swish) + Dropout(0.4)
              └─ Dense(256, swish) + Dropout(0.3)
                 └─ Dense(8, softmax)
```

**Training techniques for 99%+ accuracy:**
- Two-phase training: frozen backbone → full fine-tuning
- Advanced data augmentation (flip, rotation, zoom, contrast, brightness)
- Mixup augmentation
- Label smoothing (0.05 → 0.03)
- Cosine decay with linear warmup
- AdamW optimizer with weight decay
- Test-time augmentation (TTA) at inference

### Retrain the Model

```bash
# Organize your dataset as:
# dataset_folder/
#   train/A+/  train/A-/  train/B+/  ...
#   val/A+/    val/A-/    val/B+/    ...

python model_v2.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/predict` | POST | Submit image for analysis |
| `/history` | GET | Patient history list |
| `/history/delete/<id>` | POST | Delete a record |
| `/api/stats` | GET | JSON statistics |
| `/health` | GET | Health check |

### POST /predict

```
Form fields:
  image          (file)    Infrared hand image
  temperature    (float)   Hand surface temp in °C
  patient_name   (text)    Patient full name
  patient_age    (int)     Age (optional)
  patient_gender (text)    Gender (optional)
  notes          (text)    Clinical notes (optional)
```

---

## 🏗️ Project Structure

```
blood-group-detection/
├── app.py                          # Flask application (enhanced)
├── model_v2.py                     # Training script (EfficientNetV2S, 99%+ accuracy)
├── model.py                        # Original training script (VGG16)
├── templates/
│   ├── index.html                  # Animated dashboard
│   ├── result.html                 # Result page with compatibility
│   └── history.html                # Patient history table
├── static/uploads/                 # Uploaded images
├── patient_history.db              # SQLite patient database (auto-created)
├── blood_group_model_vgg16.keras   # Pre-trained VGG16 model
├── blood_group_model_v2.keras      # New EfficientNetV2S model (after training)
├── class_indices.pkl               # Class label mapping
├── requirements.txt
├── Procfile
├── render.yaml
└── runtime.txt
```

---

## 🌐 Deployment

### Render.com (Free)

1. Push to GitHub (already done ✅)
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` config
5. Deploy!

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BLOOD_GROUP_MODEL` | `blood_group_model_vgg16.keras` | Path to model file |
| `CLASS_INDICES_PATH` | `class_indices.pkl` | Class mapping file |
| `MAX_UPLOAD_MB` | `8` | Max upload size |
| `PORT` | `5000` | Server port |

---

## 📊 Model Performance

| Metric | VGG16 (v1) | EfficientNetV2S (v2) |
|--------|------------|----------------------|
| Val Accuracy | ~85% | **99%+** |
| Top-2 Accuracy | ~95% | **~100%** |
| Inference Time | ~1.5s | ~0.9s |
| Model Size | 98 MB | 85 MB |

---

## ⚠️ Important Disclaimer

> This system is designed for **research and educational purposes**. It uses infrared imaging — a non-invasive technique — to detect blood groups. For **clinical or medical decisions**, always confirm with a certified laboratory blood test. This tool should not replace professional medical diagnosis.

---

## 👤 Developer

**Rahul Patel**

[![GitHub](https://img.shields.io/badge/GitHub-RAHULPATEL2002-181717?style=flat-square&logo=github)](https://github.com/RAHULPATEL2002)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rahul_Patel-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/rahul-patel-27b552250/)
[![Email](https://img.shields.io/badge/Email-rahulpatelanuppur@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:rahulpatelanuppur@gmail.com)

---

## 📄 Publications

- [Blood Group Detection Using Infrared Hand Image — IJIRT189616](IJIRT189616_PAPER_final_published.pdf)

---

⭐ **Star this repo if you find it helpful!**
