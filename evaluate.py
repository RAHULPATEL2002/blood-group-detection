import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_data_dir):
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found.")
        return

    # Check if test data directory exists
    if not os.path.isdir(test_data_dir):
        print(f"❌ Test data directory '{test_data_dir}' not found.")
        return

    # Load the model
    model = load_model(model_path)

    # Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define ImageDataGenerator with rescaling
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(128, 128),
        batch_size=1,  # Use batch_size=1 for precise sample-wise evaluation
        class_mode='categorical',
        shuffle=False  # Important for evaluation and confusion matrix
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\n📊 Total Test Samples: {test_generator.samples}")
    print(f"🔢 Number of Classes: {test_generator.num_classes}")
    print(f'🎯 Test Accuracy: {accuracy * 100:.2f}%, Test Loss: {loss:.4f}')

if __name__ == '__main__':
    evaluate_model('blood_group_model_vgg16.keras', 'data/test/')
