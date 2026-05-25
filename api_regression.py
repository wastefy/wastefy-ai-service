import re
from datetime import date

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from model.schemas import (
    ErrorDetail, ErrorResponseWrapper,
    MetaInfo, SuccessResponse
)
from model.utils import get_now, load_model, verify_api_key

router = APIRouter(prefix="/predict", tags=["Regression Model"])

# 1. SCHEMA SPESIFIK REGRESSION

# Daftar Filter
VALID_ITEM = ["Anggur", "Apel", "Cabai", "Jeruk", "Kentang", "Mangga", "Mentimun", "Pisang", "Tomat", "Wortel"]
VALID_JENIS = ["Buah", "Sayur"]
VALID_KONDISI = ["Busuk", "Matang", "Mentah", "Terlalu Matang", "Segar"]
VALID_LOKASI = ["Suhu Ruang", "Pendingin", "Pembeku"]

class ModelInfo(BaseModel):
    name: str = "Regression Model"
    version: str = "1.0.0"

class InputRegresi(BaseModel):
    nama_item: str = Field(..., description="Nama item: Anggur, Apel, dll")
    jenis_item: str = Field(..., description="Jenis item: Buah, Sayur")
    lokasi_penyimpanan: str = Field(..., description="Lokasi penyimpanan: Suhu Ruang, Pendingin, Pembeku")
    tanggal_beli: str = Field(..., description="Format tanggal pembelian: YYYY-MM-DD")
    kondisi_fisik: str = Field(..., description="Kondisi fisik: Segar, Matang, Mentah, Terlalu Matang, Busuk")

    @field_validator('nama_item')
    @classmethod
    def validasi_item(cls, v):
        if v not in VALID_ITEM:
            raise ValueError(f'Item harus salah satu dari: {", ".join(VALID_ITEM)}')
        return v

    @field_validator('jenis_item')
    @classmethod
    def validasi_jenis(cls, v):
        if v not in VALID_JENIS:
            raise ValueError(f'Jenis harus: {", ".join(VALID_JENIS)}')
        return v

    @field_validator('lokasi_penyimpanan')
    @classmethod
    def validasi_lokasi(cls, v):
        if v not in VALID_LOKASI:
            raise ValueError(f'Lokasi harus: {", ".join(VALID_LOKASI)}')
        return v

    @field_validator('kondisi_fisik')
    @classmethod
    def validasi_kondisi(cls, v):
        if v not in VALID_KONDISI:
            raise ValueError(f'Kondisi harus: {", ".join(VALID_KONDISI)}')
        return v

    @field_validator('tanggal_beli')
    @classmethod
    def validasi_tanggal(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Format tanggal harus YYYY-MM-DD")
        return v

class OutputRegresi(BaseModel):
    estimasi_sisa_hari: int

# 2. LOAD MODEL & METADATA

try:
    model, metadata = load_model("regression")

    ENCODER_CLASSES   = metadata["encoder_classes"]        
    LABEL_ORDER       = metadata["label_order_mapping"]    
    LOKASI_MULT       = metadata["lokasi_multiplier"]      
    UPPER_BOUND       = metadata["upper_bound_lookup"]     
    SAFETY_CAP_RATIO  = metadata["safety_cap_ratio"]       
    SCALER_CENTER     = np.array(metadata["scaler_params"]["center_"])  
    SCALER_SCALE      = np.array(metadata["scaler_params"]["scale_"])   
except Exception as e:
    model = None
    metadata = None
    print(f"[WARNING] Gagal memuat Regression Model: {e}")


# 3. HELPER FUNCTIONS

def encode_categorical(value: str, feature_name: str) -> int:
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
    enc_item   = encode_categorical(nama_item,          "nama_item")
    enc_jenis  = encode_categorical(jenis_item,         "jenis_item")
    enc_lokasi = encode_categorical(lokasi_penyimpanan, "lokasi_penyimpanan")
    enc_label  = encode_categorical(label,              "label")

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
    ]], dtype=np.float32)

    scaled = (raw - SCALER_CENTER) / SCALER_SCALE
    return scaled

def apply_safe_prediction(pred_raw: float, nama_item: str, lokasi_penyimpanan: str) -> int:
    ub_key      = f"{nama_item}|{lokasi_penyimpanan}"
    upper_bound = UPPER_BOUND.get(ub_key, 42.0)
    safety_cap  = upper_bound * SAFETY_CAP_RATIO

    pred_clipped = float(np.clip(pred_raw, 0, safety_cap))
    return int(round(pred_clipped))


# 4. CONTOH RESPONSES

REGRESSION_RESPONSES = {
    200: {
        "model": SuccessResponse,
        "description": "Successful Response",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "code": 200,
                    "data": {
                        "estimasi_sisa_hari": 3
                    },
                    "message": "Estimasi sisa hari berhasil dihitung",
                    "meta": {
                        "api": {"version": "1.0.0"},
                        "generated_at": "2026-05-24T02:00:00Z",
                        "model": {"name": "Regression Model", "version": "1.0.0"}
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
        "description": "Validasi Gagal (Input tidak sesuai)",
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "code": 422,
                    "errors": [{"error_code": "invalid_input", "message": "Tanggal beli tidak boleh di masa depan."}],
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
                    "errors": [{"error_code": "internal_server_error", "message": "Terjadi kesalahan saat memproses prediksi..."}],
                    "message": "Terjadi kegagalan sistem",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"}
                }
            }
        }
    }
}


# 5. ENDPOINT

@router.post(
    "/regression",
    response_model=SuccessResponse,
    dependencies=[Depends(verify_api_key)],
    responses=REGRESSION_RESPONSES
)
async def prediksi_sisa_hari(data: InputRegresi):
    waktu_sekarang = get_now()
    try:
        tanggal_beli = date.fromisoformat(data.tanggal_beli)
        hari_sejak   = (date.today() - tanggal_beli).days

        if hari_sejak < 0:
            return JSONResponse(status_code=422, content=ErrorResponseWrapper(
                code=422,
                errors=[ErrorDetail(error_code="invalid_input", message="Tanggal beli tidak boleh di masa depan.")],
                message="Data tidak dapat diproses",
                meta=MetaInfo(generated_at=waktu_sekarang),
            ).model_dump())

        X = build_feature_vector(
            nama_item            = data.nama_item,
            jenis_item           = data.jenis_item,
            lokasi_penyimpanan   = data.lokasi_penyimpanan,
            label                = data.kondisi_fisik,
            hari_sejak_pembelian = hari_sejak
        )

        pred_raw = float(model.predict(X, verbose=0)[0][0])
        estimasi_hari = apply_safe_prediction(pred_raw, data.nama_item, data.lokasi_penyimpanan)

        hasil_data = OutputRegresi(estimasi_sisa_hari=estimasi_hari)

        return SuccessResponse(
            code=200,
            data=hasil_data,
            message="Estimasi sisa hari berhasil dihitung",
            meta=MetaInfo(generated_at=waktu_sekarang, model=ModelInfo().model_dump()),
        )

    except ValueError as e:
        return JSONResponse(status_code=422, content=ErrorResponseWrapper(
            code=422,
            errors=[ErrorDetail(error_code="invalid_input", message=str(e))],
            message="Data tidak dapat diproses",
            meta=MetaInfo(generated_at=waktu_sekarang),
        ).model_dump())
        
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponseWrapper(
            code=500,
            errors=[ErrorDetail(error_code="internal_server_error", message=str(e))],
            message="Terjadi kegagalan sistem",
            meta=MetaInfo(generated_at=waktu_sekarang),
        ).model_dump())