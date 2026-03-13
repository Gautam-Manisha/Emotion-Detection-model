# EmotiVision – Emotion Detection Web App

A full-stack AI web app that detects facial emotions from uploaded images or live webcam captures using a pre-trained deep learning model.

---

## 📁 Folder Structure

```
emotion-detection-webapp/
├── backend/
│   ├── app.py              # Flask API server
│   ├── model_loader.py     # Loads the .h5 Keras model
│   ├── utils.py            # Image preprocessing & bounding box
│   ├── requirements.txt    # Python dependencies
│   └── model/
│       └── emotion_model.h5  ← Place your model here
├── frontend/
│   ├── index.html          # Landing page
│   ├── detect.html         # Emotion detection page
│   ├── style.css           # Styles
│   └── script.js           # Frontend logic
├── uploads/                # Temporary uploaded images
└── README.md
```

---

## 🧠 Model

The app uses a pre-trained Keras model that:
- Accepts **grayscale images** reshaped to `(1, 48, 48, 1)`
- Outputs probabilities for **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Is based on the **FER-2013** dataset

### Where to get a model

1. **Pre-trained model** – Download from:
   - https://github.com/oarriaga/face_classification (look for `fer2013_mini_XCEPTION.102-0.66.hdf5`)
   - Kaggle FER-2013 trained models

2. **Train your own** – Use your existing `testdata.py` or any FER-2013 training script and save with:
   ```python
   model.save('emotion_model.h5')
   ```

3. **Place the file** at: `backend/model/emotion_model.h5`

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.9+
- A modern web browser

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd emotion-detection-webapp/backend

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Add the Model

Copy your `emotion_model.h5` to:
```
emotion-detection-webapp/backend/model/emotion_model.h5
```

### 3. Run the Backend

```bash
cd emotion-detection-webapp/backend
python app.py
```

The API server will start at: `http://localhost:5000`

### 4. Run the Frontend

Open the frontend directly in your browser – no build step required:

```bash
# macOS
open emotion-detection-webapp/frontend/index.html

# Or serve with Python's built-in server (avoids some browser restrictions)
cd emotion-detection-webapp/frontend
python -m http.server 8080
# Then visit http://localhost:8080
```

---

## 🔌 API Reference

### `POST /predict`

Detect emotion from a face image.

**Request:**
```
Content-Type: multipart/form-data
Body: image=<file>
```

**Success Response (200):**
```json
{
  "emotion": "Happy",
  "confidence": 0.92,
  "all_emotions": {
    "Angry": 0.01,
    "Disgust": 0.01,
    "Fear": 0.02,
    "Happy": 0.92,
    "Sad": 0.01,
    "Surprise": 0.02,
    "Neutral": 0.01
  },
  "bbox_image": "/uploads/bbox_abc123.jpg"
}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| 400    | No image file provided |
| 422    | No face detected |
| 500    | Prediction failed |

**Example using cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/face.jpg"
```

**Example using Python requests:**
```python
import requests

with open('face.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
print(response.json())
```

---

## 🖼️ Features

| Feature | Details |
|---------|---------|
| Upload image | JPG, PNG, BMP, WEBP supported |
| Drag & Drop | Drag an image onto the upload zone |
| Webcam capture | Live camera feed with one-click capture |
| Bounding box | Detected face highlighted with a green box |
| Confidence score | Animated bar showing model certainty |
| Full breakdown | All 7 emotion probabilities visualised |
| Error handling | Clear messages for no-face / server errors |
| Responsive | Works on mobile and desktop |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML | TensorFlow / Keras |
| Vision | OpenCV (Haar Cascade face detection) |
| Model | FER-2013 pre-trained CNN |

---

## ⚠️ Notes

- The Flask server must be running before using the frontend.
- Webcam capture requires HTTPS or `localhost` origin in most browsers.
- The `uploads/` folder is automatically created on first run.
- Face detection uses OpenCV's Haar Cascade – ensure the subject is well-lit and facing the camera.
