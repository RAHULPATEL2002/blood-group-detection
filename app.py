from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "blood_group_model_vgg16.keras"
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load class indices
CLASS_INDICES_PATH = "class_indices.pkl"
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "rb") as f:
        class_indices = pickle.load(f)
    class_labels = {v: k for k, v in class_indices.items()}
else:
    class_labels = {}

# Ensure the uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model not loaded. Please check server logs.", 500
    
    if 'image' not in request.files or 'temperature' not in request.form:
        return "❌ Image and temperature input are required.", 400

    img_file = request.files['image']
    temperature = request.form['temperature']

    if img_file.filename == '' or not temperature:
        return "❌ No file selected or temperature missing.", 400

    try:
        temp_value = float(temperature)
    except ValueError:
        return "❌ Temperature must be a valid number.", 400

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Append temperature as an additional feature (if model supports it)
        # You must retrain your model with temp input to use this properly.

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        blood_type = class_labels.get(class_index, "Unknown")

        # Remove uploaded file to save space
        os.remove(img_path)

        return render_template('result.html', blood_type=blood_type, temperature=temp_value)

    except Exception as e:
        return f"❌ Error processing image: {e}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)