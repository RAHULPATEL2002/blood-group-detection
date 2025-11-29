from tensorflow.keras.models import load_model
import os

def convert_keras_to_json(model_path, json_path, weights_path):
    try:
        # Load the existing .keras model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file '{model_path}' not found.")
        
        model = load_model(model_path, compile=False)

        # Convert the model architecture to JSON
        model_json = model.to_json()
        
        # Save the JSON model structure
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

        # Save the model weights separately
        if not weights_path.endswith(".weights.h5"):
            raise ValueError("❌ Weights file must end with '.weights.h5'")
        
        model.save_weights(weights_path)
        print("✅ Model successfully converted to JSON format and weights saved!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    convert_keras_to_json(
        model_path="blood_group_model_vgg16.keras",
        json_path="model.json",
        weights_path="model_weights.weights.h5"
    )
