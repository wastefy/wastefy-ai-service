import os
import io
import json
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Depends, Security, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import img_to_array

router = APIRouter(prefix="/predict", tags=["Vision Model"])

# 1. PENGAMANAN API KEY
# Mencari variabel VISION_API_KEY di .env atau server cloud.
load_dotenv()
VALID_API_KEY = os.getenv("VISION_API_KEY")
print(VALID_API_KEY)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_str: str = Security(api_key_header)):
    if api_key_str != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Akses Ditolak: API Key tidak valid atau tidak ditemukan"
        )
    return api_key_str

# 2. DEFINISI WRAPPER
class APIInfo(BaseModel):
    version: str = "1.0.0"

class ModelInfo(BaseModel):
    name: str = "Vision MobileNetV2"
    version: str = "1.0.0"

class MetaInfo(BaseModel):
    api: APIInfo = APIInfo()
    generated_at: str
    model: Optional[ModelInfo] = None

class VisionData(BaseModel):
    out_of_scope: bool
    nama_item: Optional[str] = None
    jenis_item: Optional[str] = None
    kondisi_fisik: Optional[str] = None
    confidence: float

class UnauthorizedResponse(BaseModel):
    detail: str = "Akses Ditolak: API Key tidak valid atau tidak ditemukan"

class SuccessResponse(BaseModel):
    code: int
    data: VisionData
    message: str
    meta: MetaInfo
    status: str = "success"

class ErrorDetail(BaseModel):
    error_code: str
    message: str

class ErrorResponseWrapper(BaseModel):
    code: int
    errors: List[ErrorDetail]
    message: str
    meta: MetaInfo
    status: str = "error"

# 3. LOAD MODEL & METADATA
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

# Batas maksimal ukuran file (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024 

# 4. HELPER FUNCTIONS
def compute_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def is_out_of_scope(probs: np.ndarray) -> bool:
    confidence = float(np.max(probs))
    entropy = compute_entropy(probs)
    return confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD

# 5. CONTOH RESPONSES

VISION_RESPONSES = {
    200: {
        "model": SuccessResponse,
        "description": "Successful Response",
        "content": {
            "application/json": {
                "example": {
                    "code": 200,
                    "data": {
                        "out_of_scope": False,
                        "nama_item": "Apel",
                        "jenis_item": "Buah",
                        "kondisi_fisik": "Matang",
                        "confidence": 0.9431
                    },
                    "message": "Prediksi gambar berhasil diselesaikan",
                    "meta": {
                        "api": {"version": "1.0.0"},
                        "generated_at": "2026-05-24T02:00:00Z",
                        "model": {"name": "Vision MobileNetV2", "version": "1.0.0"}
                    },
                    "status": "success"
                }
            }
        }
    },
    401: {
        "model": UnauthorizedResponse,
        "description": "Unauthorized (API Key Salah atau Tidak Ada)"
    },
    422: {
        "model": ErrorResponseWrapper,
        "description": "Validasi Gagal (Format/Ukuran File Salah)",
        "content": {
            "application/json": {
                "example": {
                    "code": 422,
                    "errors": [{"error_code": "file_too_large", "message": "Ukuran gambar melebihi batas 5MB."}],
                    "message": "Data tidak dapat diproses",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"},
                    "status": "error"
                }
            }
        }
    },
    500: {
        "model": ErrorResponseWrapper,
        "description": "Kesalahan Internal Server",
        "content": {
            "application/json": {
                "example": {
                    "code": 500,
                    "errors": [{"error_code": "internal_server_error", "message": "Terjadi kesalahan saat memproses gambar..."}],
                    "message": "Terjadi kegagalan sistem",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"},
                    "status": "error"
                }
            }
        }
    }
}

# 6. ENDPOINT
@router.post(
    "/vision",
    response_model=SuccessResponse,
    dependencies=[Depends(verify_api_key)],  # Pengunci API
    responses=VISION_RESPONSES
)
async def prediksi_gambar(file_foto: UploadFile = File(...)):
    waktu_sekarang = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Validasi Tipe File 
    if file_foto.content_type not in ["image/jpeg", "image/png"]:
        error_payload = ErrorResponseWrapper(
            code=422,
            errors=[ErrorDetail(error_code="invalid_file_format", message="Hanya mendukung file JPEG atau PNG.")],
            message="Data tidak dapat diproses",
            meta=MetaInfo(generated_at=waktu_sekarang),
        )
        return JSONResponse(status_code=422, content=error_payload.model_dump())

    try:
        contents = await file_foto.read()
        
        # Validasi Ukuran File (Maks 5MB)
        if len(contents) > MAX_FILE_SIZE:
            error_payload = ErrorResponseWrapper(
                code=422,
                errors=[ErrorDetail(error_code="file_too_large", message="Ukuran gambar melebihi batas 5MB.")],
                message="Data tidak dapat diproses",
                meta=MetaInfo(generated_at=waktu_sekarang),
            )
            return JSONResponse(status_code=422, content=error_payload.model_dump())

        # Proses Prediksi Gambar
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        predictions = model.predict(img_arr, verbose=0)[0]
        confidence = float(np.max(predictions))

        # Deteksi Out-of-Scope
        if is_out_of_scope(predictions):
            hasil_data = VisionData(
                out_of_scope=True,
                nama_item=None,
                jenis_item=None,
                kondisi_fisik=None,
                confidence=round(confidence, 4)
            )
        else:
            predicted_idx = int(np.argmax(predictions))
            parts = CLASS_NAMES[predicted_idx].split("||")
            hasil_data = VisionData(
                out_of_scope=False,
                nama_item=parts[0],
                jenis_item=parts[1],
                kondisi_fisik=parts[2],
                confidence=round(confidence, 4)
            )

        # Kembalikan Respons Sukses
        return SuccessResponse(
            code=200,
            data=hasil_data,
            message="Prediksi gambar berhasil diselesaikan",
            meta=MetaInfo(generated_at=waktu_sekarang, model=ModelInfo()),
        )

    # Tangkap Error Internal
    except Exception as e:
        error_payload = ErrorResponseWrapper(
            code=500,
            errors=[ErrorDetail(error_code="internal_server_error", message=str(e))],
            message="Terjadi kegagalan sistem saat memproses gambar",
            meta=MetaInfo(generated_at=waktu_sekarang),
        )
        return JSONResponse(status_code=500, content=error_payload.model_dump())