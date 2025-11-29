# 🩸 Blood Group Detection Using Infrared Hand Images

A Flask web application that detects blood groups from infrared hand images using deep learning (VGG16 model). This project uses computer vision and machine learning to classify blood groups based on thermal imaging of hands.

## ✨ Features

- 🔬 **Deep Learning Model**: VGG16-based CNN for accurate blood group classification
- 📸 **Image Upload**: Upload infrared hand images for analysis
- 🌡️ **Temperature Input**: Include temperature data for enhanced predictions
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🚀 **Easy Deployment**: Ready-to-deploy on Render.com or other platforms

## 📋 Project Structure

```
.
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                 # Home page with upload form
│   └── result.html                # Results display page
├── blood_group_model_vgg16.keras  # Trained VGG16 model (98MB)
├── class_indices.pkl              # Class labels mapping
├── uploads/                        # Temporary image storage
│
├── # Training & Evaluation Scripts
├── model.py                       # Model architecture definition
├── preprocess.py                  # Image preprocessing utilities
├── split_data.py                  # Dataset splitting script
├── evaluate.py                    # Model evaluation script
├── check_accuracy.py              # Accuracy checking utilities
├── overall_accuracy.py            # Overall accuracy calculation
├── training_validation_accuracy_check.py  # Training metrics
├── class_indices_pkl.py           # Generate class indices
├── convert_model.py               # Model conversion utilities
│
├── # Results & Documentation
├── confusion_matrix.png           # Model confusion matrix
├── training_validation_accuracy.png  # Training curves
├── classification_report.txt      # Detailed classification report
├── history.pkl                    # Training history
├── model.json                     # Model architecture JSON
│
├── # Documentation
├── Blood Group Detection Using InfraRed Hand Image.pdf
├── problem_statement_and_Inovation.pdf
├── DEPLOYMENT.md                  # Deployment guide
├── RENDER_SETUP.md               # Render.com setup instructions
├── README.md                      # This file
│
└── # Configuration Files
    ├── requirements.txt           # Python dependencies
    ├── Procfile                   # Deployment configuration
    ├── render.yaml                # Render.com config
    └── runtime.txt                # Python version specification
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RAHULPATEL2002/blood-group-detection.git
   cd blood-group-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Visit `http://localhost:5000`

### Production Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick Deploy to Render.com:**
1. Push your code to GitHub (already done ✅)
2. Connect your repo to [Render.com](https://render.com)
3. Render will auto-detect the configuration from `render.yaml`
4. Deploy and get your live URL!

## 📦 Requirements

- **Python**: 3.10.12 or 3.13+ (see `runtime.txt`)
- **Flask**: 3.1.0
- **TensorFlow**: 2.20.0
- **NumPy**: < 2.0.0
- **Pillow**: 10.4.0
- **Gunicorn**: 21.2.0 (for production)

## 🔧 Model Details

- **Architecture**: VGG16-based Convolutional Neural Network
- **Input Size**: 128x128 pixels
- **Classes**: 8 blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-)
- **Training**: Includes temperature-based classification

## 📊 Training Scripts

The repository includes several scripts for model training and evaluation:

- `model.py` - Define and compile the model
- `preprocess.py` - Image preprocessing pipeline
- `split_data.py` - Split dataset into train/validation/test
- `evaluate.py` - Evaluate model performance
- `check_accuracy.py` - Check model accuracy metrics
- `overall_accuracy.py` - Calculate overall accuracy
- `training_validation_accuracy_check.py` - Training metrics

## 📝 Important Notes

- ✅ Model file (`blood_group_model_vgg16.keras`) and `class_indices.pkl` are included
- ✅ Uploaded images are automatically deleted after processing
- ✅ The app uses a pre-trained VGG16 model for blood group classification
- ⚠️ Large model files (>100MB) are excluded from Git (use Git LFS if needed)

## 🌐 Live Deployment

Once deployed, your app will be available at:
`https://your-app-name.onrender.com`

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [RENDER_SETUP.md](RENDER_SETUP.md) - Render.com specific setup
- PDF files included for project documentation

## 🤝 Contributing

This project is for educational and research purposes. Feel free to fork and improve!

## 📄 License

This project is for educational/research purposes.

## 👤 Author

**Rahul Patel** - [GitHub](https://github.com/RAHULPATEL2002)

---

⭐ **Star this repo if you find it helpful!**

