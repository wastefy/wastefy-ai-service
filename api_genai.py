import os
from fastapi import APIRouter, Response
from pydantic import BaseModel
from typing import Union
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Memuat fungsi untuk membaca berkas .env
load_dotenv()

router = APIRouter(prefix="/predict", tags=["GenAI Gemini"])

# Mengambil API Key dari environment variable secara aman
api_key_gemini = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key_gemini)

# Membaca isi prompt sekali saat aplikasi berjalan
BASE_DIR = os.path.dirname(__file__)
PROMPT_PATH = os.path.join(BASE_DIR, "prompt_templates.txt")
if not os.path.exists(PROMPT_PATH):
    raise FileNotFoundError(f"File prompt tidak ditemukan: {PROMPT_PATH}")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

# Schema Input
class DataBahanBaku(BaseModel):
    nama_item: str
    jenis_item: str
    kondisi_fisik: str
    lokasi_penyimpanan: str
    sisa_hari: int

# Schema Output Sukses
class OutputPanduan(BaseModel):
    status: str
    teks_panduan: str

# Schema Output Error
class ErrorResponse(BaseModel):
    error: str

# Gunakan Union agar Swagger tahu ada dua jenis respons
@router.post("/genai", response_model=Union[OutputPanduan, ErrorResponse])
async def buat_panduan(data: DataBahanBaku, response: Response):
    try:
        prompt = PROMPT_TEMPLATE.format(
            nama_item=data.nama_item,
            jenis_item=data.jenis_item,
            kondisi_fisik=data.kondisi_fisik,
            lokasi_penyimpanan=data.lokasi_penyimpanan,
            sisa_hari=data.sisa_hari
        )
        
        gemini_response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        
        # Kembalikan menggunakan skema Pydantic (Otomatis HTTP 200 OK)
        return OutputPanduan(
            status="success",
            teks_panduan=gemini_response.text.strip()
        )
        
    except Exception as e:
        # Kembalikan error menggunakan skema Pydantic (Ubah status code menjadi 500)
        response.status_code = 500
        return ErrorResponse(error=str(e))