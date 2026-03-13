"""
Flask backend for Emotion Detection System.
Provides the /predict endpoint to accept an image and return emotion predictions.
"""

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import load_emotion_model
from utils import preprocess_image, draw_bounding_box

# ─── App Initialization ──────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup – avoids reloading on every request
print("[INFO] Loading emotion detection model...")
model = load_emotion_model()
print("[INFO] Model loaded successfully.")

# Emotion labels (must match the order used during model training)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "message": "Emotion Detection API is running."})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Accepts a multipart/form-data image file.
    Returns JSON: { "emotion": str, "confidence": float, "all_emotions": dict }
    """
    # ── 1. Validate incoming request ─────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty filename received."}), 400

    # ── 2. Save the uploaded file temporarily ─────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: jpg, jpeg, png, bmp, webp."}), 400

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # ── 3. Preprocess ──────────────────────────────────────────────────────
        face_roi, face_detected = preprocess_image(filepath)

        if not face_detected:
            return jsonify({"error": "No face detected in the image. Please upload a clear face photo."}), 422

        # ── 4. Predict ────────────────────────────────────────────────────────
        predictions = model.predict(face_roi, verbose=0)[0]  # shape: (7,)

        # Build a dict of all emotion probabilities
        all_emotions = {label: float(round(float(score), 4))
                        for label, score in zip(EMOTION_LABELS, predictions)}

        best_idx = int(predictions.argmax())
        emotion = EMOTION_LABELS[best_idx]
        confidence = float(round(float(predictions[best_idx]), 4))

        # ── 5. Optional: return bounding-box image path ───────────────────────
        bbox_filename = f"bbox_{filename}"
        bbox_path = os.path.join(UPLOAD_FOLDER, bbox_filename)
        draw_bounding_box(filepath, bbox_path)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": all_emotions,
            "bbox_image": f"/uploads/{bbox_filename}"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up the original upload (keep bbox image for display)
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve images from the uploads folder."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
