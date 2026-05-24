# 🥦 Vision Model — Klasifikasi Kondisi Fisik Sayur & Buah

Modul FastAPI untuk mengklasifikasikan kondisi fisik sayur dan buah menggunakan model **MobileNetV2** yang dilatih dengan Transfer Learning (Feature Extraction).

---

## 📁 Struktur Folder

```
model/vision/
├── .env.example            ← Template variabel lingkungan (API Key)
├── .gitignore              ← Aturan pengecualian file untuk Git
├── api_vision.py           ← Router FastAPI 
├── Model_metadata.json     ← Metadata kelas & nilai threshold
├── model.keras             ← Model Keras
├── README.md               ← Dokumentasi API
├── requirements.txt        ← Daftar dependensi Python
└── vision_model.ipynb      ← Notebook proses pelatihan
```

---

## 🧠 Spesifikasi Model

| Properti | Detail |
|---|---|
| Arsitektur | MobileNetV2 (pretrained ImageNet, frozen) |
| Input | 224 × 224 × 3 RGB |
| Jumlah Kelas | 30 |
| Dataset (total) | 68.752 gambar (24.242 dipakai setelah undersampling cap 1.500/kelas) |
| Item | Apel, Pisang, Mangga, Jeruk, Wortel, Kentang, Cabai, Anggur, Mentimun, Tomat |
| Fase Training | Feature Extraction (1 fase) |
| Augmentasi | On-the-fly (flip, crop, brightness, dll.) |

### Label Output

Format internal label dari model: `nama_item||jenis_item||kondisi_fisik`

| `jenis_item` | `kondisi_fisik` |
|---|---|
| Buah | Mentah, Matang, Terlalu Matang, Busuk |
| Sayur | Segar, Busuk |

---

## 🚀 Spesifikasi Endpoint

### `POST /predict/vision`

Menerima file gambar untuk dianalisis dan mengembalikan prediksi kondisi fisiknya dalam format JSON terstruktur.

#### Aturan Validasi

- **Format:** Hanya menerima `image/jpeg` atau `image/png`
- **Ukuran:** Maksimal 5 MB

#### Request Headers

```http
X-API-Key: <kunci_rahasia_api_anda>
Content-Type: multipart/form-data
```

#### Request Body

| Field | Tipe | Keterangan |
|---|---|---|
| `file_foto` | File | Wajib. File fisik gambar yang akan dianalisis |

---

## 📥 Contoh Respons API

### ✅ 200 OK, Gambar Dikenali

```json
{
  "code": 200,
  "data": {
    "out_of_scope": false,
    "nama_item": "Apel",
    "jenis_item": "Buah",
    "kondisi_fisik": "Matang",
    "confidence": 0.9431
  },
  "message": "Prediksi gambar berhasil diselesaikan",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": { "name": "Vision MobileNetV2", "version": "1.0.0" }
  },
  "status": "success"
}
```

### ✅ 200 OK, Out of Scope (Gambar tidak sesuai dataset)

```json
{
  "code": 200,
  "data": {
    "out_of_scope": true,
    "nama_item": null,
    "jenis_item": null,
    "kondisi_fisik": null,
    "confidence": 0.1823
  },
  "message": "Prediksi gambar berhasil diselesaikan",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": { "name": "Vision MobileNetV2", "version": "1.0.0" }
  },
  "status": "success"
}
```

### ❌ 401 Unauthorized (API Key Salah atau Tidak Ada)

```json
{
  "code": 401,
  "errors": [
    {
      "error_code": "unauthorized",
      "message": "API Key tidak valid"
    }
  ],
  "message": "Akses ditolak",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": null
  },
  "status": "error"
}
```

### ❌ 422 Unprocessable Entity (Format/Ukuran File Salah)

```json
{
  "code": 422,
  "errors": [
    {
      "error_code": "file_too_large",
      "message": "Ukuran gambar melebihi batas 5MB."
    }
  ],
  "message": "Data tidak dapat diproses",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z"
  },
  "status": "error"
}
```

### ❌ 500 Internal Server Error (Sistem Gagal)

```json
{
  "code": 500,
  "errors": [
    {
      "error_code": "internal_server_error",
      "message": "Terjadi kesalahan saat memproses gambar..."
    }
  ],
  "message": "Terjadi kegagalan sistem",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z"
  },
  "status": "error"
}
```

---

## Cara Pakai

### 1. Install Dependensi

Pastikan kamu menggunakan virtual environment (opsional namun disarankan).

```bash
pip install -r requirements.txt
```

### 2. Atur Variabel Lingkungan

Gunakan file `.env.example` sebagai referensi. Buat salinannya dan beri nama `.env` di direktori yang sama, lalu masukkan kunci rahasia milikmu:
```bash
# Pengguna Linux/Mac
cp .env.example .env

# Pengguna Windows (Command Prompt)
copy .env.example .env
```

Pastikan isi file `.env` terlihat seperti ini:
```env 
VISION_API_KEY=kunci_rahasia_123
```

### 3. Daftarkan Router ke `main.py`

```python
from model.vision.api_vision import router as api_vision

app.include_router(api_vision)
```

### 4. Jalankan Server

**Untuk Pengujian Lokal (Development):**  
Gunakan opsi `--reload` agar server otomatis menyala ulang saat ada perubahan kode.
```bash
uvicorn main:app --reload
```

**Untuk Production (Hugging Face Spaces / Cloud):**  
Wajib mendefinisikan host dan port secara eksplisit tanpa fitur reload agar performa server tetap stabil.
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 5. Uji Endpoint via cURL

```bash
curl -X POST "http://localhost:8000/predict/vision" \
     -H "X-API-Key: kunci_rahasia_123" \
     -F "file_foto=@contoh_apel.jpg"
```

---

## Out-of-Scope Detection

Untuk mencegah model menebak secara acak pada gambar yang bukan buah atau sayur, sistem dilengkapi dengan dua filter penolakan:

| Metrik Validasi | Nilai Default | Logika Penolakan |
|---|---|---|
| Confidence Threshold | `0.60` | Ditolak jika skor tertinggi < threshold |
| Entropy Threshold | `1.134` | Ditolak jika probabilitas terlalu merata (bias) |

Nilai ini tersimpan di `model_metadata.json` dan dibaca otomatis saat startup.

---

## Training

Notebook pelatihan lengkap: [`vision_model.ipynb`](vision_model.ipynb)
