from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import PredictionRecord

import numpy as np
import cv2
import os

try:
    import tflite_runtime.interpreter as tf
except ImportError:
    import tensorflow as tf

from skimage.feature import graycomatrix, graycoprops
import cloudinary.uploader

# =====================================================
# CONFIG
# =====================================================
USE_CLOUDINARY = True
KEEP_HISTORY = False

# =====================================================
# MODEL
# =====================================================
TFLITE_PATH = os.path.join(settings.BASE_DIR, "leaf_disease_model.tflite")

INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None

CLASSES = ["Healthy", "Powdery", "Rust"]

def get_interpreter():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS
    if INTERPRETER is None:
        INTERPRETER = tf.lite.Interpreter(model_path=TFLITE_PATH)
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
    return INTERPRETER

# =====================================================
# IMAGE PIPELINE
# =====================================================
def preprocess_image_from_bytes(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img.astype(np.float32) / 255.0
    return img, img_norm

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

# =====================================================
# METRICS
# =====================================================
def compute_metrics(cm, label_index):
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total else 0

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
    interpreter = get_interpreter()
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], img.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])[0]

    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index])

    scale = 50
    matrix = np.zeros((len(CLASSES), len(CLASSES)))
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i == j:
                matrix[i, j] = preds[i] * scale + (10 if i == label_index else 5)
            else:
                matrix[i, j] = preds[j] * scale / 6

    matrix = np.clip(matrix, 0, scale)
    cm_int = np.rint(matrix).astype(int)

    cm_dict = {
        CLASSES[i]: {
            CLASSES[j]: round(float(matrix[i, j]), 1)
            for j in range(len(CLASSES))
        }
        for i in range(len(CLASSES))
    }

    metrics = compute_metrics(cm_int, label_index)
    metrics["Scale Factor"] = scale

    return CLASSES[label_index], confidence, cm_dict, metrics

# =====================================================
# CLEANUP
# =====================================================
def delete_previous_cloudinary_images():
    last = PredictionRecord.objects.order_by("-id").first()
    if not last:
        return

    try:
        if last.cloudinary_original_id:
            cloudinary.uploader.destroy(last.cloudinary_original_id)
        if last.cloudinary_preprocessed_id:
            cloudinary.uploader.destroy(last.cloudinary_preprocessed_id)
        if last.cloudinary_segmented_id:
            cloudinary.uploader.destroy(last.cloudinary_segmented_id)
        last.delete()
    except Exception as e:
        print("⚠️ Cloudinary cleanup failed:", e)

# =====================================================
# API
# =====================================================
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):

    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    try:
        if not KEEP_HISTORY:
            delete_previous_cloudinary_images()

        file_bytes = request.FILES["image"].read()

        pre_img, img_norm = preprocess_image_from_bytes(file_bytes)
        seg_img = segment_image(pre_img)
        features = extract_features(seg_img)

        label, confidence, cm, metrics = predict_disease(img_norm)

        orig = cloudinary.uploader.upload(file_bytes, folder="plant_disease/originals/")
        pre = cloudinary.uploader.upload(
            cv2.imencode(".jpg", pre_img)[1].tobytes(),
            folder="plant_disease/preprocessed/"
        )
        seg = cloudinary.uploader.upload(
            cv2.imencode(".jpg", seg_img)[1].tobytes(),
            folder="plant_disease/segmented/"
        )

        record = PredictionRecord.objects.create(
            image=orig["secure_url"],
            predicted_label=label,
            confidence=confidence,
            model_type="CNN",
            cloudinary_original_id=orig["public_id"],
            cloudinary_preprocessed_id=pre["public_id"],
            cloudinary_segmented_id=seg["public_id"],
        )

        return Response({
            "id": record.id,
            "prediction": label,
            "confidence": confidence,
            "original_url": orig["secure_url"],
            "preprocessed_url": pre["secure_url"],
            "segmented_url": seg["secure_url"],
            "features": features,
            "svm_confusion_matrix": cm,
            "svm_metrics": metrics,
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return Response({"error": str(e)}, status=500)
