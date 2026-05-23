# Core Foundation (Shared Utilities & Schemas)

Tiga file ini adalah **fondasi bersama** yang digunakan oleh seluruh modul API (Vision, GenAI, dll.). Letakkan di `model/` sebelum mengembangkan modul apapun.

---

## Struktur File

```
model/
├── config.py       ← Manajemen environment variable & konfigurasi aplikasi
├── utils.py        ← Dependency autentikasi & helper waktu
└── schemas.py      ← Skema input/output Pydantic yang dipakai bersama
```

---

## `config.py` (Konfigurasi Aplikasi)

Membaca dan memvalidasi environment variable dari file `.env` menggunakan `pydantic-settings`. Aplikasi akan **langsung crash saat startup** jika ada variabel yang tidak ditemukan — mencegah bug yang baru muncul saat runtime.

### Variabel yang Diperlukan

| Variabel | Keterangan |
|---|---|
| `GEMINI_API_KEY` | API Key untuk Google Gemini (dipakai modul GenAI) |
| `VISION_API_KEY` | API Key untuk proteksi endpoint Vision & GenAI |

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

## 📄 `schemas.py` (Skema Input & Output)

Mendefinisikan seluruh model Pydantic yang dipakai bersama antar modul. Terbagi menjadi dua kelompok besar: **Input** dan **Output**.

### Input Schemas

#### `DataBahanBaku`

Dipakai oleh endpoint `POST /predict/genai`. Semua field divalidasi ketat — nilai di luar daftar berikut akan ditolak dengan error 422.

| Field | Tipe | Nilai Valid |
|---|---|---|
| `nama_item` | `str` | `Anggur`, `Apel`, `Cabai`, `Jeruk`, `Kentang`, `Mangga`, `Mentimun`, `Pisang`, `Tomat`, `Wortel` |
| `jenis_item` | `str` | `Buah`, `Sayur` |
| `kondisi_fisik` | `str` | `Busuk`, `Matang`, `Mentah`, `Terlalu Matang`, `Segar` |
| `lokasi_penyimpanan` | `str` | `Suhu Ruang`, `Pendingin`, `Pembeku` |
| `sisa_hari` | `int` | `0` s/d `365` |

### Output Schemas

Semua endpoint menggunakan envelope respons yang seragam.

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

#### `ErrorResponseWrapper` — HTTP 4xx / 5xx

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
    "generated_at": "2026-05-24T02:00:00Z"
  }
}
```

#### `UnauthorizedResponse` — HTTP 401

```json
{
  "detail": "Akses Ditolak: API Key tidak valid atau tidak ditemukan"
}
```

---

## Cara Pakai di Modul Baru

```python
from fastapi import APIRouter, Depends, Response
from model.utils import verify_api_key, get_now
from model.schemas import SuccessResponse, ErrorResponseWrapper, MetaInfo

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/predict/contoh", response_model=SuccessResponse[dict])
async def contoh_endpoint(response: Response):
    return SuccessResponse(
        message="Berhasil",
        data={"hasil": "..."},
        meta=MetaInfo(generated_at=get_now(), model={"name": "MyModel", "version": "1.0.0"})
    )
```

---

## Dependensi

Pastikan sudah terinstall sebelum menjalankan aplikasi:

```bash
pip install -r requirements.txt
```