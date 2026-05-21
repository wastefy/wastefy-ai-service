# 🥦 Vision Model — Klasifikasi Kesegaran Sayur & Buah

Modul FastAPI untuk mengklasifikasikan kesegaran sayur dan buah menggunakan model **MobileNetV2** yang dilatih dengan Transfer Learning (Feature Extraction).

---

## Struktur Folder

```
model/vision/
├── api_vision.py               ← Router FastAPI (file ini)
├── sayur_buah_classifier.keras ← Model Keras (download dari Google Colab)
├── model_metadata.json         ← Metadata kelas & threshold
├── requirements.txt
└── README.md
```

---

## Model

| Properti         | Detail                                      |
|------------------|---------------------------------------------|
| Arsitektur       | MobileNetV2 (pretrained ImageNet, frozen)   |
| Input            | 224 × 224 × 3                               |
| Jumlah kelas     | 30                                          |
| Dataset (total)  | 68.752 gambar (24.242 dipakai setelah undersampling cap 1.500/kelas) |
| Item             | apple, banana, carrot, orange, tomat, cabe, timun, potato, grape, mango |
| Fase training    | Feature Extraction (1 fase)                 |
| Augmentasi       | On-the-fly (flip, crop, brightness, dll.)   |

### Label Output

Format internal label: `nama_item||jenis_item||kondisi_fisik`

| `jenis_item` | `kondisi_fisik`                |
|--------------|--------------------------------------------|
| Buah         | Mentah, Matang, Terlalu Matang, Busuk      |
| Sayur        | Segar, Busuk                               |

---

## Endpoint

### `POST /predict/vision`

Menerima gambar dan mengembalikan klasifikasi kesegaran.

**Request:**

```
Content-Type: multipart/form-data
file_foto: <file gambar JPG/PNG>
```

**Response (berhasil):**

```json
{
  "out_of_scope"  : false,
  "nama_item"     : "apple",
  "jenis_item"    : "Buah",
  "kondisi_fisik" : "Matang",
  "confidence"    : 0.9431
}
```

**Response (gambar tidak dikenali):**

```json
{
  "out_of_scope" : true,
  "confidence"   : 0.1823,
  "pesan"        : "Gambar tidak dikenali sebagai sayur atau buah yang diketahui."
}
```

---

## Cara Pakai

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Pastikan file model tersedia

File berikut sudah di-commit ke repo dan harus ada di folder yang sama:

| File                            | Keterangan                         |
|---------------------------------|------------------------------------|
| `sayur_buah_classifier.keras`   | Bobot model Keras (13 MB)         |
| `model_metadata.json`           | Nama kelas & nilai threshold       |

### 3. Daftarkan router di `main.py`

```python
from model.vision.api_vision import router as vision_router

app.include_router(vision_router)
```

### 4. Jalankan server

```bash
uvicorn main:app --reload
```

### 5. Uji endpoint

```bash
curl -X POST "http://localhost:8000/vision/predict" \
     -F "file_foto=@contoh_apel.jpg"
```

---

## Out-of-Scope Detection

Model dilengkapi dua mekanisme untuk menolak gambar di luar dataset:

| Mekanisme           | Nilai default | Keterangan                                      |
|---------------------|---------------|-------------------------------------------------|
| Confidence threshold | 0.60          | Tolak jika skor tertinggi < threshold           |
| Entropy threshold    | 2.80          | Tolak jika distribusi probabilitas terlalu merata |

Nilai ini tersimpan di `model_metadata.json` dan dibaca otomatis saat startup.

---

## Training

Notebook pelatihan lengkap: [`sayur_buah_classifier.ipynb`](../../notebooks/sayur_buah_classifier.ipynb)
