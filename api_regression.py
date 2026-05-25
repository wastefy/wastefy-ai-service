# api_regression.py
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import date
import tensorflow as tf
import numpy as np
import json
import os

router = APIRouter(prefix="/predict", tags=["Regression Model"])


# ── 1. Pydantic Models ────────────────────────────────────────────────────────

class InputRegresi(BaseModel):
    nama_item: str
    jenis_item: str
    lokasi_penyimpanan: str
    tanggal_beli: str   # format: "YYYY-MM-DD"
    kondisi_fisik: str  # label kondisi: Segar, Matang, Mentah, Terlalu Matang, Busuk

class OutputRegresi(BaseModel):
    estimasi_sisa_hari: int

class ErrorResponse(BaseModel):
    error: str


# ── 2. Load Model & Metadata ──────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(BASE_DIR, "model.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    model = None
    print(f"[WARNING] Gagal load model: {e}")

try:
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    # Struktur metadata sesuai notebook Cell 10
    ENCODER_CLASSES   = metadata["encoder_classes"]        # dict: col -> [class list]
    LABEL_ORDER       = metadata["label_order_mapping"]    # {"Busuk":0, ..., "Mentah":4}
    LOKASI_MULT       = metadata["lokasi_multiplier"]      # {"Suhu Ruang":1.0, ...}
    UPPER_BOUND       = metadata["upper_bound_lookup"]     # {"Apel|Pembeku": 42.0, ...}
    SAFETY_CAP_RATIO  = metadata["safety_cap_ratio"]       # 0.92
    SCALER_CENTER     = np.array(metadata["scaler_params"]["center_"])  # shape (9,)
    SCALER_SCALE      = np.array(metadata["scaler_params"]["scale_"])   # shape (9,)

except Exception as e:
    metadata = None
    print(f"[WARNING] Gagal load metadata: {e}")


# ── 3. Helper: Preprocessing (harus identik dengan Cell 4 & 5 notebook) ───────

def encode_categorical(value: str, feature_name: str) -> int:
    """Label encode satu nilai kategorikal menggunakan classes dari metadata."""
    classes = ENCODER_CLASSES[feature_name]
    if value not in classes:
        raise ValueError(
            f"Nilai '{value}' tidak dikenal untuk fitur '{feature_name}'. "
            f"Nilai valid: {classes}"
        )
    return classes.index(value)


def build_feature_vector(
    nama_item: str,
    jenis_item: str,
    lokasi_penyimpanan: str,
    label: str,
    hari_sejak_pembelian: int
) -> np.ndarray:

    # Encode kategorikal
    enc_item   = encode_categorical(nama_item,          "nama_item")
    enc_jenis  = encode_categorical(jenis_item,         "jenis_item")
    enc_lokasi = encode_categorical(lokasi_penyimpanan, "lokasi_penyimpanan")
    enc_label  = encode_categorical(label,              "label")

    # Fitur turunan (feature engineering, sama persis dengan notebook Cell 4)
    label_score    = LABEL_ORDER.get(label, 2)
    lokasi_mult    = LOKASI_MULT.get(lokasi_penyimpanan, 1.0)
    label_x_lokasi = label_score * lokasi_mult
    hari_x_lokasi  = hari_sejak_pembelian * lokasi_mult

    raw = np.array([[
        enc_item,
        enc_jenis,
        enc_lokasi,
        enc_label,
        hari_sejak_pembelian,
        label_score,
        lokasi_mult,
        label_x_lokasi,
        hari_x_lokasi
    ]], dtype=np.float32)  # shape (1, 9)

    # Scaling dengan RobustScaler params dari metadata (X_scaled = (X - center) / scale)
    scaled = (raw - SCALER_CENTER) / SCALER_SCALE
    return scaled


def apply_safe_prediction(pred_raw: float, nama_item: str, lokasi_penyimpanan: str) -> int:

    ub_key      = f"{nama_item}|{lokasi_penyimpanan}"
    upper_bound = UPPER_BOUND.get(ub_key, 42.0)
    safety_cap  = upper_bound * SAFETY_CAP_RATIO

    pred_clipped = float(np.clip(pred_raw, 0, safety_cap))
    return int(round(pred_clipped))


# ── 4. Endpoint ───────────────────────────────────────────────────────────────

@router.post("/regression", response_model=OutputRegresi)
async def prediksi_sisa_hari(data: InputRegresi):
    if model is None or metadata is None:
        return ErrorResponse(error="Model atau metadata tidak ditemukan.")

    try:
        # Hitung hari sejak pembelian
        tanggal_beli = date.fromisoformat(data.tanggal_beli)
        hari_sejak   = (date.today() - tanggal_beli).days

        if hari_sejak < 0:
            return ErrorResponse(error="Tanggal beli tidak boleh di masa depan.")

        # Preprocessing: buat feature vector 9 dimensi + scaling
        X = build_feature_vector(
            nama_item            = data.nama_item,
            jenis_item           = data.jenis_item,
            lokasi_penyimpanan   = data.lokasi_penyimpanan,
            label                = data.kondisi_fisik,   # 'kondisi_fisik' -> 'label'
            hari_sejak_pembelian = hari_sejak
        )

        # Prediksi dengan model
        pred_raw = float(model.predict(X, verbose=0)[0][0])

        # Safe prediction: clip + integer
        estimasi_hari = apply_safe_prediction(pred_raw, data.nama_item, data.lokasi_penyimpanan)

        return OutputRegresi(estimasi_sisa_hari=estimasi_hari)

    except ValueError as e:
        # Input tidak dikenal (nama item / lokasi / label tidak ada di training data)
        return ErrorResponse(error=str(e))
    except Exception as e:
        return ErrorResponse(error=f"Terjadi kesalahan: {str(e)}")
