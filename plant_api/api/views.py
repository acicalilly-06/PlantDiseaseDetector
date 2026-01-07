from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import PredictionRecord
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser

import numpy as np
import cv2
import os
from uuid import uuid4
from skimage.feature import graycomatrix, graycoprops

# =====================================================
# FLAGS
# =====================================================

USE_CLOUDINARY = (
    not settings.DEBUG and
    hasattr(settings, "DEFAULT_FILE_STORAGE") and
    "cloudinary_storage" in settings.DEFAULT_FILE_STORAGE
)

if USE_CLOUDINARY:
    import cloudinary.uploader  # type: ignore

KEEP_HISTORY = False  # HARD RESET EACH TIME (Render-safe)

# =====================================================
# MODEL
# =====================================================
# api/views.py
_model = None

def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(
            os.path.join(settings.BASE_DIR, "ml", "leaf_disease_model.h5")
        )
    return _model

CLASSES = ["Healthy", "Powdery", "Rust"]

# =====================================================
# IMAGE PIPELINE
# =====================================================

def preprocess_image_from_bytes(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

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
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
    }

    mean_color = cv2.mean(segmented_image)[:3]
    features.update({
        "mean_R": round(mean_color[2], 2),
        "mean_G": round(mean_color[1], 2),
        "mean_B": round(mean_color[0], 2),
    })

    return {k: round(v, 3) for k, v in features.items()}

# =====================================================
# CONFUSION MATRIX (SYNTHETIC, SINGLE IMAGE)
# =====================================================

def compute_metrics(cm, label_index):
    cm = cm.astype(int)
    total = int(np.sum(cm))
    correct = int(np.trace(cm))
    acc = correct / total if total else 0

    TP = int(cm[label_index, label_index])
    FP = int(np.sum(cm[:, label_index]) - TP)
    FN = int(np.sum(cm[label_index, :]) - TP)
    TN = int(total - (TP + FP + FN))

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def predict_disease(img_norm):
    # ✅ Lazy-load model (NO global load)
    model = get_model()

    img = np.expand_dims(img_norm, axis=0)
    preds = model.predict(img, verbose=0)[0]

    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index])

    scale = 50
    raw = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i == j:
                raw[i, j] = preds[i] * scale
            else:
                raw[i, j] = preds[j] * scale / 6

    raw = np.clip(raw, 0, scale)
    cm_int = np.rint(raw).astype(int)

    confusion_matrix = {
        CLASSES[i]: {
            CLASSES[j]: int(cm_int[i, j])
            for j in range(len(CLASSES))
        }
        for i in range(len(CLASSES))
    }

    metrics = compute_metrics(cm_int, label_index)

    return CLASSES[label_index], confidence, confusion_matrix, metrics

# =====================================================
# API
# =====================================================

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    try:
        image_file = request.FILES["image"]
        file_bytes = image_file.read()

        pre_img, img_norm = preprocess_image_from_bytes(file_bytes)
        seg_img = segment_image(pre_img)
        features = extract_features(seg_img)

        label, confidence, confusion_matrix, metrics = predict_disease(img_norm)

        # ===== HARD RESET (Render safe) =====
        if not USE_CLOUDINARY:
            predictions_dir = os.path.join(settings.MEDIA_ROOT, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)

            for f in os.listdir(predictions_dir):
                fp = os.path.join(predictions_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)

            if not KEEP_HISTORY:
                PredictionRecord.objects.all().delete()

            uid = uuid4().hex
            orig = f"{uid}.jpg"
            pre = f"pre_{uid}.jpg"
            seg = f"seg_{uid}.jpg"

            with open(os.path.join(predictions_dir, orig), "wb") as f:
                f.write(file_bytes)

            cv2.imwrite(os.path.join(predictions_dir, pre), pre_img)
            cv2.imwrite(os.path.join(predictions_dir, seg), seg_img)

            original_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{orig}")
            pre_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{pre}")
            seg_url = request.build_absolute_uri(settings.MEDIA_URL + f"predictions/{seg}")

        else:
            original_url = cloudinary.uploader.upload(file_bytes)["secure_url"]
            pre_url = cloudinary.uploader.upload(cv2.imencode(".jpg", pre_img)[1].tobytes())["secure_url"]
            seg_url = cloudinary.uploader.upload(cv2.imencode(".jpg", seg_img)[1].tobytes())["secure_url"]

        record = PredictionRecord.objects.create(
            image=original_url,
            predicted_label=label,
            confidence=confidence,
            model_type="CNN",
        )

        return Response({
    "id": record.id,
    "prediction": label,
    "confidence": confidence,
    "original_url": original_url,
    "preprocessed_url": pre_url,
    "segmented_url": seg_url,
    "features": features,
    "svm_confusion_matrix": confusion_matrix,  # ✅ FIX
    "svm_metrics": metrics,
})

    except Exception as e:
        print("❌ Prediction error:", e)
        return Response({"error": "Prediction failed"}, status=500)
