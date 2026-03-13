"""
model_loader.py
Loads the pre-trained Keras/TensorFlow emotion detection model from disk.
The model is expected at: backend/model/emotion_model.h5
"""

import os
import sys

# Attempt to import TensorFlow; provide a helpful error message if not installed.
try:
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError:
    print("[ERROR] TensorFlow is not installed. Run: pip install tensorflow")
    sys.exit(1)

# Path to the model file (relative to this file's location)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'emotion_model.h5')


def load_emotion_model():
    """
    Load and return the emotion detection Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist at MODEL_PATH.
        RuntimeError: If the model fails to load for any other reason.

    Returns:
        keras.Model: The compiled emotion detection model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Model file not found at: {MODEL_PATH}\n"
            "Please place your emotion_model.h5 file inside the 'backend/model/' directory."
        )

    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"[INFO] Model loaded from: {MODEL_PATH}")
        return model
    except Exception as exc:
        raise RuntimeError(f"[ERROR] Failed to load model: {exc}") from exc
