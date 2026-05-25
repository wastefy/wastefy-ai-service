# Regression (Estimasi Sisa Hari Sayuran & Buah)

Model deep learning untuk memprediksi **sisa hari ketahanan** sayuran dan buah-buahan berdasarkan jenis item, kondisi fisik, dan lokasi penyimpanan. Dibangun sebagai bagian dari sistem rekomendasi kesegaran bahan makanan.

---

## Deskripsi Proyek

Proyek ini merupakan komponen **AI/ML** dari tim yang terdiri dari:

- AI Engineer — membangun dan melatih model prediksi
- Fullstack Developer — membangun antarmuka dan integrasi sistem
- Data Scientist — eksplorasi data dan analisis fitur

Model menerima input berupa nama item, jenis, lokasi penyimpanan, tanggal beli, dan kondisi fisik, lalu mengembalikan estimasi sisa hari dalam bentuk **integer** dengan mekanisme _safe prediction_ untuk mencegah overprediksi berbahaya.

---

## Struktur File

```
├── regression_model.ipynb      # Notebook training lengkap (11 cell)
├── api_regression.py           # FastAPI router endpoint prediksi
├── model.keras                 # Model terlatih (dihasilkan setelah run notebook)
├── model_metadata.json         # Encoder, scaler, upper-bound lookup (dihasilkan setelah run notebook)
└── README.md
```

---

## Arsitektur Model

Model menggunakan **Feedforward Deep Neural Network** dengan arsitektur:

```
Input (9 fitur)
    │
Dense(256) → BatchNorm → ReLU → Dropout(0.25)
    │
Dense(128) → BatchNorm → ReLU → Dropout(0.25)
    │
Dense(64)  → BatchNorm → ReLU
    │
Dense(32)  → ReLU
    │
Dense(1)   → Output (sisa hari)
```

**Loss function:** Huber loss (robust terhadap outlier)  
**Optimizer:** Adam (lr=1e-3, dengan ReduceLROnPlateau)

---

## Input

Model dilatih dengan **9 fitur** hasil feature engineering:

| #   | Fitur                  | Tipe        | Keterangan                                                      |
| --- | ---------------------- | ----------- | --------------------------------------------------------------- |
| 1   | `nama_item`            | Kategorikal | Nama sayuran/buah                                               |
| 2   | `jenis_item`           | Kategorikal | `Sayur` / `Buah`                                                |
| 3   | `lokasi_penyimpanan`   | Kategorikal | `Suhu Ruang` / `Pendingin` / `Pembeku`                          |
| 4   | `label`                | Kategorikal | Kondisi: `Busuk`, `Terlalu Matang`, `Matang`, `Segar`, `Mentah` |
| 5   | `hari_sejak_pembelian` | Numerik     | Dihitung dari `tanggal_beli` ke hari ini                        |
| 6   | `label_score`          | Derived     | Ordinal encoding kondisi (0–4)                                  |
| 7   | `lokasi_mult`          | Derived     | Bobot ketahanan per lokasi (1.0 / 1.8 / 3.5)                    |
| 8   | `label_x_lokasi`       | Interaksi   | `label_score × lokasi_mult`                                     |
| 9   | `hari_x_lokasi`        | Interaksi   | `hari_sejak_pembelian × lokasi_mult`                            |

**Item yang didukung:** Anggur, Apel, Cabe, Jeruk, Kentang, Mangga, Pisang, Timun, Tomat, Wortel

---

## Mekanisme Safe Prediction

Untuk mencegah overprediksi berbahaya (model memprediksi item masih segar padahal sudah mendekati busuk), setiap prediksi melewati tiga lapisan pengamanan:

1. **Clip ke 0** — tidak ada prediksi negatif
2. **Upper-bound clipping** — prediksi tidak boleh melebihi nilai maksimum historis untuk kombinasi `nama_item + lokasi_penyimpanan` dari data training
3. **Safety buffer 8%** — prediksi di-cap di 92% upper bound sebagai margin keamanan tambahan
4. **Pembulatan ke integer** — sisa hari dikembalikan sebagai bilangan bulat

```
Contoh:
  Tomat di Suhu Ruang → upper bound historis = 12 hari
  Safety cap (92%)    = 11 hari
  Jika model prediksi 15 → dikembalikan 11
```

---

## Struktur Notebook

| Cell | Isi                                                         |
| ---- | ----------------------------------------------------------- |
| 1    | Import library & konfigurasi path                           |
| 2    | Load data & eksplorasi awal                                 |
| 3    | Analisis noise floor & pembuatan upper-bound lookup table   |
| 4    | Feature engineering & augmentasi data (3× lipat)            |
| 5    | Label encoding & RobustScaler (params disimpan di metadata) |
| 6    | Definisi arsitektur model deep learning                     |
| 7    | Konfigurasi callbacks & training                            |
| 8    | Evaluasi model & fungsi `safe_predict_batch`                |
| 9    | Analisis MAE & rekomendasi peningkatan                      |
| 10   | Simpan `model.keras` & `model_metadata.json`                |
| 11   | Contoh inference end-to-end                                 |

**Callbacks yang digunakan:**

- `EarlyStopping` — berhenti jika `val_mae` tidak membaik selama 30 epoch
- `ReduceLROnPlateau` — turunkan learning rate 50% saat stagnan selama 12 epoch
- `ModelCheckpoint` — simpan bobot terbaik secara otomatis

---

## Cara Pakai

### 1. Install Dependensi

```bash
pip install tensorflow scikit-learn pandas numpy jupyter fastapi uvicorn
```

Versi yang direkomendasikan:

```
tensorflow >= 2.12
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
```

### 2. Daftarkan Router ke `main.py`

```python
from fastapi import FastAPI
from api_regression import router as regression_router

app = FastAPI()
app.include_router(regression_router)
```

### 3. Jalankan Server

```bash
uvicorn main:app --reload
```

### 4. Uji Endpoint via cURL

```bash
curl -X POST "http://localhost:8000/predict/regression" \
     -H "Content-Type: application/json" \
     -d '{
       "nama_item": "Tomat",
       "jenis_item": "Sayur",
       "kondisi_fisik": "Segar"
       "lokasi_penyimpanan": "Pendingin",
       "tanggal_beli": "2026-05-10",
     }'
```

---

## Spesifikasi Endpoint

```
POST /predict/regression
```

### Request Body

**Headers:**
- `X-API-Key`: (Wajib) API Key untuk autentikasi.

| Field                | Tipe     | Keterangan                                                  |
| -------------------- | -------- | ----------------------------------------------------------- |
| `nama_item`          | `string` | Nama item (hanya yang didukung, misal: Anggur, Apel, dll)   |
| `jenis_item`         | `string` | `Buah` atau `Sayur`                                         |
| `kondisi_fisik`      | `string` | `Segar`, `Matang`, `Mentah`, `Terlalu Matang`, atau `Busuk` |
| `lokasi_penyimpanan` | `string` | `Suhu Ruang`, `Pendingin`, atau `Pembeku`                   |
| `tanggal_beli`       | `string` | Format `YYYY-MM-DD`, tidak boleh di masa depan              |

```json
{
  "nama_item": "Tomat",
  "jenis_item": "Sayur",
  "kondisi_fisik": "Segar"
  "lokasi_penyimpanan": "Pendingin",
  "tanggal_beli": "2026-05-10",
}

## Contoh Response API

### ✅ 200 OK — Prediksi Berhasil

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "estimasi_sisa_hari": 7
  },
  "message": "Estimasi sisa hari berhasil dihitung",
  "meta": {
    "api": {
      "version": "1.0.0"
    },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": {
      "name": "Regression Model",
      "version": "1.0.0"
    }
  }
}
```

### ❌ 401 Unauthorized — API Key Tidak Valid

Dikembalikan jika header X-API-Key salah atau tidak disertakan.

```json
{
  "status": "error",
  "code": 401,
  "errors": [
    {
      "error_code": "unauthorized",
      "message": "API Key tidak valid"
    }
  ],
  "message": "Akses ditolak",
  "meta": {
    "api": {
      "version": "1.0.0"
    },
    "generated_at": "2026-05-24T02:00:00Z"
  }
}
```

### ❌ 422 Unprocessable Entity — Validasi Input Gagal

Dikembalikan otomatis jika format salah, item tidak terdaftar (Pydantic validation), atau tanggal di masa depan.

```json
{
  "status": "error",
  "code": 422,
  "errors": [
    {
      "error_code": "invalid_input",
      "message": "Item harus salah satu dari: Anggur, Apel, Cabai..." 
    }
  ],
  "message": "Data tidak dapat diproses",
  "meta": {
    "api": {
      "version": "1.0.0"
    },
    "generated_at": "2026-05-24T02:00:00Z"
  }
}
```

### ❌ 500 Internal Server Error — Kesalahan Sistem

Dikembalikan jika model tidak ditemukan (gagal dimuat) atau terjadi crash saat inferensi.

```json
{
  "status": "error",
  "code": 500,
  "errors": [
    {
      "error_code": "internal_server_error",
      "message": "Model atau metadata tidak ditemukan."
    }
  ],
  "message": "Terjadi kegagalan sistem",
  "meta": {
    "api": {
      "version": "1.0.0"
    },
    "generated_at": "2026-05-24T02:00:00Z"
  }
}
```

---

## Performa Model

| Metrik                | Nilai      |
| --------------------- | ---------- |
| MAE (raw, test set)   | ~2.28 hari |
| MAE (safe prediction) | ~2.3 hari  |
| RMSE                  | ~3.36 hari |