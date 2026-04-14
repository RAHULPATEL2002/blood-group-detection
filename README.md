# Infrared Blood Group Detection

Infrared Blood Group Detection is a Flask web app that classifies blood groups from infrared hand images using a VGG16-based deep learning model. The UI is mobile-first, supports camera capture, and returns confidence-ranked predictions in seconds.

## Features

- Mobile-friendly upload flow with camera capture support
- Infrared hand image preview and guidance tips
- Temperature input with range validation
- Confidence breakdown for top predicted blood groups
- Auto-cleanup of uploaded images (older than 24 hours)
- Production-ready health check endpoint

## Project Structure

```
.
├── app.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── uploads/ (runtime files)
├── blood_group_model_vgg16.keras
├── class_indices.pkl
├── requirements.txt
├── render.yaml
└── runtime.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

## Deployment Notes

- Render/Heroku-style deployment uses `gunicorn app:app`.
- For mobile camera capture, deploy with HTTPS (Render provides HTTPS automatically).
- Model files are required on the server:
  - `blood_group_model_vgg16.keras`
  - `class_indices.pkl`

## Health Check

`GET /health`

Example:

```json
{
  "status": "ok",
  "model_ready": true
}
```

## Disclaimer

This project is for research and educational use only. Confirm all blood group results using clinical laboratory testing.
