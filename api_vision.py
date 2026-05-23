from fastapi import APIRouter, UploadFile, File, Response
from pydantic import BaseModel
from typing import Optional, Union
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from tensorflow.keras.preprocessing.image import img_to_array

router = APIRouter(prefix="/predict", tags=["Vision Model"])

# 1. Definisi Pydantic Model (Output Schema)
class OutputVision(BaseModel):
    out_of_scope: bool
    nama_item: Optional[str] = None
    jenis_item: Optional[str] = None
    kondisi_fisik: Optional[str] = None
    confidence: float

# Buat schema khusus untuk error
class ErrorResponse(BaseModel):
    error: str

# 2. Load model & metadata
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"File metadata tidak ditemukan: {METADATA_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]
CONFIDENCE_THRESHOLD = metadata.get("confidence_threshold", 0.6)
ENTROPY_THRESHOLD = metadata.get("entropy_threshold", 1.134)

IMG_SIZE = 224

# 3. Helper Functions
def compute_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def is_out_of_scope(probs: np.ndarray) -> bool:
    confidence = float(np.max(probs))
    entropy = compute_entropy(probs)
    return confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD

# 4. Endpoint 
# bisa merespons OutputVision ATAU ErrorResponse
@router.post("/vision", response_model=Union[OutputVision, ErrorResponse])
async def prediksi_gambar(response: Response, file_foto: UploadFile = File(...)):
    if file_foto.content_type not in ["image/jpeg", "image/png"]:
        response.status_code = 400
        return ErrorResponse(error="Hanya mendukung file JPEG atau PNG.")

    try:
        contents = await file_foto.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        predictions = model.predict(img_arr, verbose=0)[0]
        confidence = float(np.max(predictions))

        # Deteksi Out-of-scope 
        if is_out_of_scope(predictions):
            return OutputVision(
                out_of_scope=True,
                nama_item=None,
                jenis_item=None,
                kondisi_fisik=None,
                confidence=round(confidence, 4)
            )

        # Proses Label
        predicted_idx = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[predicted_idx]
        parts = predicted_label.split("||")

        return OutputVision(
            out_of_scope=False,
            nama_item=parts[0],
            jenis_item=parts[1],
            kondisi_fisik=parts[2],
            confidence=round(confidence, 4)
        )

    except Exception as e:
        response.status_code = 500
        return ErrorResponse(error=f"Terjadi kesalahan saat memproses gambar: {str(e)}")