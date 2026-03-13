"""
utils.py
Image preprocessing helpers for the emotion detection backend.
- Detects faces using OpenCV Haar Cascade.
- Resizes / normalises the face ROI for model input.
- Optionally draws a bounding box on the original image.
"""

import cv2
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

# Model input size (must match training configuration)
IMG_SIZE = 48  # Most FER-2013 models expect 48×48 grayscale input

# OpenCV's pre-trained Haar Cascade for frontal face detection
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess_image(image_path: str):
    """
    Load an image, detect a face, extract the ROI and prepare it for the model.

    Args:
        image_path (str): Absolute path to the saved image file.

    Returns:
        Tuple[np.ndarray, bool]:
            - face_array: 4-D numpy array ready for model.predict()  (1, 48, 48, 1)
            - face_detected: True if at least one face was found, False otherwise.
    """
    # Read image from disk
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None, False

    # Use the largest detected face (by area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Extract and preprocess the face ROI
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_roi = face_roi.astype('float32') / 255.0          # Normalize to [0, 1]
    face_roi = np.expand_dims(face_roi, axis=-1)             # (48, 48, 1)
    face_roi = np.expand_dims(face_roi, axis=0)              # (1, 48, 48, 1)

    return face_roi, True


def draw_bounding_box(source_path: str, dest_path: str) -> None:
    """
    Draw a green bounding box around the detected face and save to dest_path.

    Args:
        source_path (str): Path to the original uploaded image.
        dest_path (str): Path where the annotated image will be saved.
    """
    img = cv2.imread(source_path)
    if img is None:
        return  # Silently skip if image cannot be read

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 100), 3)

    cv2.imwrite(dest_path, img)
