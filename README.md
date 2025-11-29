# Blood Group Detection App

A Flask web application that detects blood groups from infrared hand images using deep learning (VGG16 model).

## Features

- Upload infrared hand images
- Enter temperature data
- Get blood group predictions
- Modern, responsive UI

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and visit: `http://localhost:5000`

### Production Deployment

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick Deploy to Render.com:**
1. Push your code to GitHub
2. Connect your repo to Render.com
3. Use the provided `render.yaml` configuration
4. Deploy!

## Project Structure

```
.
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   └── result.html                # Results page
├── blood_group_model_vgg16.keras  # Trained model
├── class_indices.pkl              # Class labels mapping
├── uploads/                       # Temporary image storage
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment configuration
├── render.yaml                    # Render.com config
└── DEPLOYMENT.md                  # Detailed deployment guide
```

## Requirements

- Python 3.10+
- Flask 3.1.0
- TensorFlow 2.18.0
- NumPy 2.0.2
- Pillow 10.4.0

## Notes

- Model files (`blood_group_model_vgg16.keras` and `class_indices.pkl`) must be present for the app to work
- Uploaded images are automatically deleted after processing
- The app uses a VGG16-based CNN model for blood group classification

## License

This project is for educational/research purposes.

