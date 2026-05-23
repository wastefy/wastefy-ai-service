# api_regression.py
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import date
import tensorflow as tf
import numpy as np
import json
import os

router = APIRouter(prefix="/predict", tags=["Regression Model"])

# 1. Definisi Pydantic Model
class InputRegresi(BaseModel):
    nama_item: str
    jenis_item: str
    lokasi_penyimpanan: str
    tanggal_beli: str
    kondisi_fisik: str

class OutputRegresi(BaseModel):
    estimasi_sisa_hari: int

class ErrorResponse(BaseModel):
    error: str

# 2. Load model & metadata
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception:
    model = None

try:
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    # Ambil encoder classes dari metadata
    ENCODER_CLASSES = {}
    for cat in metadata["features"]["categorical_features"]:
        ENCODER_CLASSES[cat["name"]] = cat["encoder_classes"]
    SCALER_MEAN = metadata["features"]["numeric_features"][0]["scaler_mean"]
    SCALER_SCALE = metadata["features"]["numeric_features"][0]["scaler_scale"]
except Exception:
    ENCODER_CLASSES = {}
    SCALER_MEAN = 0
    SCALER_SCALE = 1

# 3. Helper Functions
def encode(value, feature_name):
    return ENCODER_CLASSES[feature_name].index(value)

def scale(value):
    return (value - SCALER_MEAN) / SCALER_SCALE

# 4. Endpoint
@router.post("/regression")
async def prediksi_sisa_hari(data: InputRegresi):
    if model is None:
        return ErrorResponse(error="Model tidak ditemukan.")
    
    try:
        # Hitung hari sejak pembelian
        tanggal_beli = date.fromisoformat(data.tanggal_beli)
        hari_sejak = (date.today() - tanggal_beli).days
        
        if hari_sejak < 0:
            return ErrorResponse(error="Tanggal tidak valid")
        
        # Preprocessing
        input_array = np.array([[
            encode(data.nama_item, "nama_item"),
            encode(data.jenis_item, "jenis_item"),
            encode(data.lokasi_penyimpanan, "lokasi_penyimpanan"),
            encode(data.kondisi_fisik, "label"),
            scale(hari_sejak)
        ]], dtype=np.float32)
        
        # Prediksi
        pred = model.predict(input_array, verbose=0)[0][0]
        estimasi_hari = int(max(0, round(pred)))
        
        return OutputRegresi(estimasi_sisa_hari=estimasi_hari)
    
    except Exception as e:
        return ErrorResponse(error=str(e))