import io
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.preprocessing.image import img_to_array

from model.schemas import (
    ErrorDetail, ErrorResponseWrapper,
    MetaInfo, SuccessResponse
)
from model.utils import get_now, load_model, verify_api_key

router = APIRouter(prefix="/predict", tags=["Vision Model"])

# 1. SCHEMA SPESIFIK VISION
class ModelInfo(BaseModel):
    name: str = "Vision MobileNetV2"
    version: str = "1.0.0"

class VisionData(BaseModel):
    out_of_scope: bool
    nama_item: Optional[str] = None
    jenis_item: Optional[str] = None
    kondisi_fisik: Optional[str] = None
    confidence: float

# 2. LOAD MODEL & METADATA
try:
    model, metadata = load_model("vision")
    
    CLASS_NAMES = metadata["class_names"]
    CONFIDENCE_THRESHOLD = metadata.get("confidence_threshold", 0.6)
    ENTROPY_THRESHOLD = metadata.get("entropy_threshold", 1.134)
except Exception as e:
    model = None
    metadata = None
    print(f"[WARNING] Gagal memuat Vision Model: {e}")

CLASS_NAMES = metadata["class_names"]
CONFIDENCE_THRESHOLD = metadata.get("confidence_threshold", 0.6)
ENTROPY_THRESHOLD = metadata.get("entropy_threshold", 1.134)
IMG_SIZE = 224
MAX_FILE_SIZE = 5 * 1024 * 1024

# 3. HELPER FUNCTIONS
def compute_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def is_out_of_scope(probs: np.ndarray) -> bool:
    confidence = float(np.max(probs))
    entropy = compute_entropy(probs)
    return confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD

# 4. CONTOH RESPONSES
VISION_RESPONSES = {
    200: {
        "model": SuccessResponse,
        "description": "Successful Response",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
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
                    }
                }
            }
        }
    },
    401: {
        "model": ErrorResponseWrapper,
        "description": "Unauthorized (API Key Salah atau Tidak Ada)",
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "code": 401,
                    "errors": [{"error_code": "unauthorized", "message": "API Key tidak valid"}],
                    "message": "Akses ditolak",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"}
                }
            }
        }
    },
    422: {
        "model": ErrorResponseWrapper,
        "description": "Validasi Gagal (Format/Ukuran File Salah)",
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "code": 422,
                    "errors": [{"error_code": "file_too_large", "message": "Ukuran gambar melebihi batas 5MB."}],
                    "message": "Data tidak dapat diproses",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"}
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
                    "status": "error",
                    "code": 500,
                    "errors": [{"error_code": "internal_server_error", "message": "Terjadi kesalahan saat memproses gambar..."}],
                    "message": "Terjadi kegagalan sistem",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"}
                }
            }
        }
    }
}

# 5. ENDPOINT
@router.post(
    "/vision",
    response_model=SuccessResponse,
    dependencies=[Depends(verify_api_key)],
    responses=VISION_RESPONSES
)
async def prediksi_gambar(file_foto: UploadFile = File(..., description="File gambar JPEG atau PNG, maksimal 5MB")):

    if file_foto.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=422, content=ErrorResponseWrapper(
            code=422,
            errors=[ErrorDetail(error_code="invalid_file_format", message="Hanya mendukung file JPEG atau PNG.")],
            message="Data tidak dapat diproses",
            meta=MetaInfo(generated_at=get_now()),
        ).model_dump())

    try:
        contents = await file_foto.read()

        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(status_code=422, content=ErrorResponseWrapper(
                code=422,
                errors=[ErrorDetail(error_code="file_too_large", message="Ukuran gambar melebihi batas 5MB.")],
                message="Data tidak dapat diproses",
                meta=MetaInfo(generated_at=get_now()),
            ).model_dump())

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        predictions = model.predict(img_arr, verbose=0)[0]
        confidence = float(np.max(predictions))

        if is_out_of_scope(predictions):
            hasil_data = VisionData(out_of_scope=True, confidence=round(confidence, 4))
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

        return SuccessResponse(
            code=200,
            data=hasil_data,
            message="Prediksi gambar berhasil diselesaikan",
            meta=MetaInfo(generated_at=get_now(), model=ModelInfo().model_dump()),
        )

    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponseWrapper(
            code=500,
            errors=[ErrorDetail(error_code="internal_server_error", message=str(e))],
            message="Terjadi kegagalan sistem",
            meta=MetaInfo(generated_at=get_now()),
        ).model_dump())