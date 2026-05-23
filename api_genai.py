import os
from fastapi import APIRouter, Depends
from google import genai
from google.genai import types

# Impor dari folder model
from model.config import settings
from model.utils import verify_api_key, get_now
from model.schemas import (
    DataBahanBaku, MetaInfo, UnauthorizedResponse,
    SuccessResponse, ErrorDetail, ErrorResponseWrapper
)

router = APIRouter(prefix="/predict", tags=["GenAI Gemini"])

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
        "model": UnauthorizedResponse,
        "description": "Unauthorized (API Key Salah atau Tidak Ada)"
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
                    "errors": [{"error_code": "genai_error", "message": "Gagal terhubung ke Gemini API"}],
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
            data={"teks_panduan": gemini_response.text.strip()},
            message="Panduan berhasil dibuat",
            meta=MetaInfo(generated_at=waktu_sekarang)
        )
        
    except Exception as e:
        from fastapi.responses import JSONResponse
        error_payload = ErrorResponseWrapper(
            code=500,
            errors=[ErrorDetail(error_code="genai_error", message=str(e))],
            message="Terjadi kesalahan saat memproses GenAI",
            meta=MetaInfo(generated_at=waktu_sekarang)
        )
        return JSONResponse(status_code=500, content=error_payload.model_dump())