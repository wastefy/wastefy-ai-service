# Core Foundation (Shared Utilities & Schemas)

Tiga file ini adalah **fondasi bersama** yang digunakan oleh seluruh modul API (Vision, GenAI, dll.). Letakkan di `model/` sebelum mengembangkan modul apapun.

---

## Struktur File

```
model/
├── config.py       ← Manajemen environment variable & konfigurasi aplikasi
├── utils.py        ← Dependency autentikasi & helper waktu
└── schemas.py      ← Skema output Pydantic yang dipakai bersama
```

---

## `config.py` (Konfigurasi Aplikasi)

Membaca dan memvalidasi environment variable dari file `.env` menggunakan `pydantic-settings`. Aplikasi akan **langsung crash saat startup** jika ada variabel yang tidak ditemukan — mencegah bug yang baru muncul saat runtime.

### Variabel yang Diperlukan

| Variabel | Keterangan |
|---|---|
| `GEMINI_API_KEY` | API Key untuk Google Gemini (dipakai modul GenAI) |
| `VISION_API_KEY` | API Key untuk proteksi endpoint |

### Penggunaan

```python
from model.config import settings

api_key = settings.GEMINI_API_KEY
```

### Contoh `.env`

```env
GEMINI_API_KEY=your_gemini_api_key_here
VISION_API_KEY=your_secret_api_key_here
```

---

## `utils.py` (Helper & Dependency Autentikasi)

Menyediakan dua fungsi yang dipakai di seluruh router API.

### `get_now()`

Mengembalikan timestamp UTC saat ini dalam format ISO 8601 untuk field `generated_at` di setiap respons.

```python
from model.utils import get_now

get_now()  # → "2026-05-24T02:00:00Z"
```

### `verify_api_key()`

FastAPI dependency untuk memvalidasi header `X-API-Key` di setiap request. Jika key tidak valid atau tidak ada, otomatis mengembalikan **401 Unauthorized**.

```python
from fastapi import APIRouter, Depends
from model.utils import verify_api_key

router = APIRouter()

@router.post("/endpoint", dependencies=[Depends(verify_api_key)])
async def my_endpoint():
    ...
```

---

## `schemas.py` (Skema Output Bersama)

Mendefinisikan model Pydantic untuk envelope respons yang dipakai seragam di seluruh modul. Schema input yang spesifik per modul (seperti `DataBahanBaku`) didefinisikan di file API masing-masing.

### Output Schemas

#### `SuccessResponse[T]` — HTTP 200

Generic schema untuk respons sukses. `T` diisi dengan tipe data spesifik tiap modul.

```json
{
  "code": 200,
  "status": "success",
  "message": "...",
  "data": { ... },
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": { "name": "...", "version": "..." }
  }
}
```

> Field `meta.model` hanya diisi untuk endpoint yang menggunakan model lokal (contoh: Vision). Untuk endpoint lain nilainya `null`.

#### `ErrorResponseWrapper` — HTTP 4xx / 5xx

Dipakai untuk semua respons error termasuk 401, 422, dan 500.

```json
{
  "code": 422,
  "status": "error",
  "message": "Data tidak dapat diproses",
  "errors": [
    {
      "error_code": "validation_error",
      "message": "..."
    }
  ],
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": null
  }
}
```

---

## Cara Pakai di Modul Baru

```python
from fastapi import APIRouter, Depends
from model.utils import verify_api_key, get_now
from model.schemas import SuccessResponse, ErrorResponseWrapper, MetaInfo

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/predict/contoh", response_model=SuccessResponse[dict])
async def contoh_endpoint():
    return SuccessResponse(
        message="Berhasil",
        data={"hasil": "..."},
        meta=MetaInfo(generated_at=get_now())
    )
```

---

## Dependensi

Pastikan sudah terinstall sebelum menjalankan aplikasi:

```bash
pip install -r requirements.txt
```
