from pathlib import Path

import tensorflow as tf


SOURCE_MODEL = Path("blood_group_model_efficientnet.keras")
TARGET_MODEL = Path("blood_group_model_efficientnet.tflite")


def convert_keras_to_tflite(source_model: Path = SOURCE_MODEL, target_model: Path = TARGET_MODEL) -> None:
    if not source_model.exists():
        raise FileNotFoundError(f"Model file not found: {source_model}")

    model = tf.keras.models.load_model(source_model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    target_model.write_bytes(tflite_model)
    print(f"Saved TFLite model to {target_model} ({target_model.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    convert_keras_to_tflite()
