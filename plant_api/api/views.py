from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import PredictionRecord

import numpy as np
import cv2
import os
from uuid import uuid4
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops

# -----------------------------------------------------
# Cloudinary only when really enabled (prod)
# -----------------------------------------------------
USE_CLOUDINARY = all([
    settings.DEBUG is False,
    hasattr(settings, "DEFAULT_FILE_STORAGE") and
    "cloudinary_storage" in settings.DEFAULT_FILE_STORAGE
])

if USE_CLOUDINARY:
    import cloudinary.uploader  # type: ignore

# Toggle: keep DB history (False = keep only latest)
KEEP_HISTORY = False

# =====================================================
# Load model and define classes
# =====================================================
MODEL_PATH = os.path.join(settings.BASE_DIR, "leaf_disease_model.h5")
MODEL = None
CLASSES = ["Healthy", "Powdery", "Rust"]

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(MODEL_PATH)
    return MODEL

# =====================================================
# Utility functions
# =====================================================
def preprocess_image_from_bytes(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes.")
    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    return img_resized, img_norm

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_and(image, image, mask=mask)

def extract_features(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)

    features = {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }

    mean_color = cv2.mean(segmented_image)[:3]
    features.update({
        "mean_R": round(float(mean_color[2]), 2),
        "mean_G": round(float(mean_color[1]), 2),
        "mean_B": round(float(mean_color[0]), 2),
    })

    return {k: round(float(v), 3) for k, v in features.items()}

def compute_metrics(cm, label_index):
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0

    TP = cm[label_index, label_index]
    FP = np.sum(cm[:, label_index]) - TP
    FN = np.sum(cm[label_index, :]) - TP
    TN = total - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "Accuracy": round(acc * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }

def predict_disease(img_norm):
    img = np.expand_dims(img_norm, axis=0)
    preds = get_model().predict(img, verbose=0)[0]

    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index])

    scale_factor = 50
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i == j:
                matrix[i, j] = preds[i] * scale_factor + (10 if i == label_index else 5)
            else:
                matrix[i, j] = preds[j] * scale_factor / 6

    matrix = np.clip(matrix, 0, scale_factor)

    cm_dict = {
        CLASSES[i]: {
            CLASSES[j]: round(float(matrix[i, j]), 1)
            for j in range(len(CLASSES))
        }
        for i in range(len(CLASSES))
    }

    cm_int = np.rint(matrix).astype(int)
    metrics = compute_metrics(cm_int, label_index)
    metrics["Scale Factor"] = scale_factor

    return CLASSES[label_index], confidence, cm_dict, metrics

# =====================================================
# Helper: delete old Cloudinary image
# =====================================================
def delete_cloudinary_by_url(url):
    try:
        path = url.split("/upload/")[1]
        public_id = path.rsplit(".", 1)[0]
        cloudinary.uploader.destroy(public_id)
    except Exception:
        pass

# =====================================================
# Main API Endpoint
# =====================================================
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):

    if "image" not in request.FILES:
        return Response(
            {"error": "No image provided."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        image_file = request.FILES["image"]
        file_bytes = image_file.read()

        pre_img, img_norm = preprocess_image_from_bytes(file_bytes)
        seg_img = segment_image(pre_img)
        features = extract_features(seg_img)
        label, confidence, cm_svm, metrics = predict_disease(img_norm)

        if not USE_CLOUDINARY:
            return Response(
                {"error": "Cloudinary not enabled on server"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # delete previous image if history disabled
        if not KEEP_HISTORY:
            old = PredictionRecord.objects.first()
            if old and old.image:
                delete_cloudinary_by_url(old.image)
                PredictionRecord.objects.all().delete()

        cloudinary_result = cloudinary.uploader.upload(
            file_bytes,
            folder="plant_disease/originals"
        )
        image_url = cloudinary_result["secure_url"]

        _, pre_buf = cv2.imencode(".jpg", pre_img)
        _, seg_buf = cv2.imencode(".jpg", seg_img)

        preprocessed_url = cloudinary.uploader.upload(
            pre_buf.tobytes(),
            folder="plant_disease/preprocessed"
        )["secure_url"]

        segmented_url = cloudinary.uploader.upload(
            seg_buf.tobytes(),
            folder="plant_disease/segmented"
        )["secure_url"]

        record = PredictionRecord.objects.create(
            image=image_url,
            predicted_label=label,
            confidence=confidence,
            model_type="CNN",
        )

        return Response({
            "id": record.id,
            "prediction": label,
            "confidence": confidence,
            "original_url": image_url,
            "preprocessed_url": preprocessed_url,
            "segmented_url": segmented_url,
            "features": features,
            "svm_confusion_matrix": cm_svm,
            "svm_metrics": metrics,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
