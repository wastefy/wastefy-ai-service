import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from model.config import settings
from model.utils import verify_api_key, get_now
from model.schemas import (
    MetaInfo, SuccessResponse, 
    ErrorDetail, ErrorResponseWrapper
)

router = APIRouter(prefix="/predict", tags=["GenAI Gemini"])

# INPUT SCHEMA
VALID_ITEM = ["Anggur", "Apel", "Cabai", "Jeruk", "Kentang", "Mangga", "Mentimun", "Pisang", "Tomat", "Wortel"]
VALID_JENIS = ["Buah", "Sayur"]
VALID_KONDISI = ["Busuk", "Matang", "Mentah", "Terlalu Matang", "Segar"]
VALID_LOKASI = ["Suhu Ruang", "Pendingin", "Pembeku"]

class DataBahanBaku(BaseModel):
    nama_item: str
    jenis_item: str
    kondisi_fisik: str
    lokasi_penyimpanan: str
    sisa_hari: int = Field(..., ge=0, le=40)

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

    @field_validator('kondisi_fisik')
    @classmethod
    def validasi_kondisi(cls, v):
        if v not in VALID_KONDISI:
            raise ValueError(f'Kondisi harus: {", ".join(VALID_KONDISI)}')
        return v

    @field_validator('lokasi_penyimpanan')
    @classmethod
    def validasi_lokasi(cls, v):
        if v not in VALID_LOKASI:
            raise ValueError(f'Lokasi harus: {", ".join(VALID_LOKASI)}')
        return v

# Client & Prompt Setup
client = genai.Client(api_key=settings.GEMINI_API_KEY)

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.txt")
if not os.path.exists(PROMPT_PATH):
    raise FileNotFoundError(f"File prompt tidak ditemukan: {PROMPT_PATH}")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

# Contoh Responses
GENAI_RESPONSES = {
    200: {
        "model": SuccessResponse[dict],
        "description": "Successful Response",
        "content": {
            "application/json": {
                "example": {
                    "code": 200,
                    "data": {"cara_simpan": "Simpan di suhu ruang dan hindari sinar matahari langsung..."},
                    "message": "Panduan berhasil dibuat",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"},
                    "status": "success"
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
                    "code": 401,
                    "errors": [{"error_code": "unauthorized", "message": "API Key tidak valid"}],
                    "message": "Akses ditolak",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"},
                    "status": "error"
                }
            }
        }
    },
    422: {
        "model": ErrorResponseWrapper,
        "description": "Validasi Gagal (Input Tidak Sesuai Schema)",
        "content": {
            "application/json": {
                "example": {
                    "code": 422,
                    "errors": [{"error_code": "validation_error", "message": "Item tidak dikenal."}],
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
                    "errors": [{"error_code": "internal_server_error", "message": "Gagal terhubung ke Gemini API"}],
                    "message": "Terjadi kegagalan sistem",
                    "meta": {"api": {"version": "1.0.0"}, "generated_at": "2026-05-24T02:00:00Z"},
                    "status": "error"
                }
            }
        }
    }
}

# Endpoint GenAI
@router.post(
    "/genai",
    response_model=SuccessResponse[dict],
    dependencies=[Depends(verify_api_key)],
    responses=GENAI_RESPONSES
)
async def buat_panduan(data: DataBahanBaku):
    waktu_sekarang = get_now()
    try:
        status_kondisi = "Kedaluwarsa/Perlu Segera Diolah" if data.sisa_hari == 0 else data.kondisi_fisik

        prompt = PROMPT_TEMPLATE.format(
            nama_item=data.nama_item, jenis_item=data.jenis_item,
            kondisi_fisik=status_kondisi, lokasi_penyimpanan=data.lokasi_penyimpanan,
            sisa_hari=data.sisa_hari
        )

        gemini_response = client.models.generate_content(
            model="gemini-3.5-flash", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )

        return SuccessResponse(
            data={"cara_simpan": gemini_response.text.strip()},
            message="Panduan berhasil dibuat",
            meta=MetaInfo(generated_at=waktu_sekarang)
        )

    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponseWrapper(
            code=500,
            errors=[ErrorDetail(error_code="genai_error", message=str(e))],
            message="Terjadi kesalahan saat memproses GenAI",
            meta=MetaInfo(generated_at=waktu_sekarang)
        ).model_dump())