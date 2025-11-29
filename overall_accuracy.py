import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Constants
model_path = "blood_group_model_vgg16.keras"
base_dir = r"D:\pythonimageprocessing\data"
target_size = (128, 128)
batch_size = 32
exclude_class = "O Negative"

# Load model
model = load_model(model_path)

# Create ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Function to load data
def load_generator(data_path, classes=None):
    return datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=classes  # None = all classes
    )

# Evaluation function
def evaluate(model, generator):
    Y_pred = model.predict(generator, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes
    acc = accuracy_score(y_true, y_pred)
    return acc * 100

# Get class names from training folder
all_classes = sorted([
    d for d in os.listdir(os.path.join(base_dir, 'train'))
    if os.path.isdir(os.path.join(base_dir, 'train', d))
])

# Create class list excluding "O Negative"
filtered_classes = [cls for cls in all_classes if cls != exclude_class]

# Helper to print evaluation
def run_evaluation(name, class_list):
    print(f"\n📊 Evaluating: {name}")
    train_gen = load_generator(os.path.join(base_dir, "train"), class_list)
    val_gen = load_generator(os.path.join(base_dir, "validation"), class_list)
    test_gen = load_generator(os.path.join(base_dir, "test"), class_list)

    print(f"✔️ Train Accuracy     : {evaluate(model, train_gen):.2f}%")
    print(f"✔️ Validation Accuracy: {evaluate(model, val_gen):.2f}%")
    print(f"✔️ Test Accuracy      : {evaluate(model, test_gen):.2f}%")

# 📌 Evaluate with all 8 classes
run_evaluation("With All Classes", all_classes)

# 📌 Evaluate after removing "O Negative"
run_evaluation("Without 'O Negative'", filtered_classes)
