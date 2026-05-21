from fastapi import APIRouter, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

router = APIRouter(prefix="/predict", tags=["Vision Model"])

# Load model & metadata
BASE_DIR = os.path.dirname(__file__)

try:
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "sayur_buah_classifier.keras")
    )
except Exception:
    model = None

try:
    with open(os.path.join(BASE_DIR, "model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    CLASS_NAMES          = metadata["class_names"]
    CONFIDENCE_THRESHOLD = metadata.get("confidence_threshold", 0.60)
    ENTROPY_THRESHOLD    = metadata.get("entropy_threshold", 2.80)
except Exception:
    CLASS_NAMES          = []
    CONFIDENCE_THRESHOLD = 0.60
    ENTROPY_THRESHOLD    = 2.80

IMG_SIZE = 224


# Helper 
def compute_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def is_out_of_scope(probs: np.ndarray) -> bool:
    confidence = float(np.max(probs))
    entropy    = compute_entropy(probs)
    return confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD


# Endpoint
@router.post("/vision")
async def prediksi_gambar(file_foto: UploadFile = File(...)):
    """
    Klasifikasi kesegaran sayur & buah.

    Returns:
        out_of_scope  : true jika gambar di luar dataset yang dikenal
        nama_item     : nama item (misal 'apple', 'tomat')
        jenis_item    : 'Buah' atau 'Sayur'
        kondisi_fisik : 'Mentah' | 'Matang' | 'Terlalu Matang' | 'Busuk' | 'Segar'
        confidence    : skor kepercayaan model (0.0 – 1.0)
    """
    if model is None:
        return {"error": "Model tidak ditemukan. Pastikan 'sayur_buah_classifier.keras' ada di direktori yang sama."}

    if not CLASS_NAMES:
        return {"error": "Metadata kelas tidak ditemukan. Pastikan 'model_metadata.json' tersedia."}

    try:
        contents  = await file_foto.read()
        img       = Image.open(io.BytesIO(contents)).convert("RGB")
        img       = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)[0]
        confidence  = float(np.max(predictions))

        # Out-of-scope detection
        if is_out_of_scope(predictions):
            return {
                "out_of_scope" : True,
                "confidence"   : round(confidence, 4),
                "pesan"        : "Gambar tidak dikenali sebagai sayur atau buah yang diketahui.",
            }

        predicted_idx   = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[predicted_idx]

        # Format label: "nama_item||jenis_item||kondisi_fisik"
        parts = predicted_label.split("||")
        nama_item, jenis_item, kondisi_fisik = parts[0], parts[1], parts[2]

        return {
            "out_of_scope" : False,
            "nama_item"    : nama_item,
            "jenis_item"   : jenis_item,
            "kondisi_fisik": kondisi_fisik,
            "confidence"   : round(confidence, 4),
        }

    except Exception as e:
        return {"error": str(e)}