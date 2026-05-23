from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, TypeVar, Generic

# INPUT SCHEMAS

## GENAI
### Daftar valid untuk filter
VALID_ITEM = ["Anggur", "Apel", "Cabai", "Jeruk", "Kentang", "Mangga", "Mentimun", "Pisang", "Tomat", "Wortel"]
VALID_JENIS = ["Buah", "Sayur"]
VALID_KONDISI = ["Busuk", "Matang", "Mentah", "Terlalu Matang", "Segar"]
VALID_LOKASI = ["Suhu Ruang", "Pendingin", "Pembeku"]

class DataBahanBaku(BaseModel):
    nama_item: str
    jenis_item: str
    kondisi_fisik: str
    lokasi_penyimpanan: str
    sisa_hari: int = Field(..., ge=0, le=365)

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

# OUTPUT SCHEMAS

## Tipe data generik untuk respons
T = TypeVar("T")

class MetaInfo(BaseModel):
    api: dict = {"version": "1.0.0"}
    generated_at: str
    model: Optional[dict] = None

class UnauthorizedResponse(BaseModel):
    detail: str = "Akses Ditolak: API Key tidak valid atau tidak ditemukan"
    
class SuccessResponse(BaseModel, Generic[T]):
    code: int = 200
    data: T
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