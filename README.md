# Vision Model — Klasifikasi Kondisi Fisik Sayur & Buah

Modul FastAPI untuk mengklasifikasikan kondisi fisik sayur dan buah menggunakan model **MobileNetV2** yang dilatih dengan Transfer Learning (Feature Extraction).

---

## Struktur Folder

```
model/vision/
├── api_vision.py               ← Router FastAPI 
├── model_metadata.json         ← Metadata kelas & threshold
├── model.keras                 ← Model Keras
├── README.md                   ← Dokumentasi
├── requirements.txt            ← Daftar dependensi
└── vision_model.ipynb         ← Notebook
```

---

## Model

| Properti         | Detail                                      |
|------------------|---------------------------------------------|
| Arsitektur       | MobileNetV2 (pretrained ImageNet, frozen)   |
| Input            | 224 × 224 × 3                               |
| Jumlah kelas     | 30                                          |
| Dataset (total)  | 68.752 gambar (24.242 dipakai setelah undersampling cap 1.500/kelas) |
| Item             | Apel, Pisang, Mangga, Jeruk, Wortel, Kentang, Cabai, Anggur, Mentimun, Tomat |
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

Menerima gambar dan mengembalikan klasifikasi kondisi fisik.

**Request:**

```
Content-Type: multipart/form-data
file_foto: <file gambar JPG/PNG>
```

**Response (berhasil):**

```json
{
  "out_of_scope"  : false,
  "nama_item"     : "Apel",
  "jenis_item"    : "Buah",
  "kondisi_fisik" : "Matang",
  "confidence"    : 0.9431
}
```

**Response (gambar tidak dikenali):**

```json
{
  "out_of_scope" : true,
  "nama_item"    : null,
  "jenis_item"   : null,
  "kondisi_fisik": null,
  "confidence"   : 0.1823
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
| `model.keras`   | Bobot model Keras (13 MB)         |
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
curl -X POST "http://localhost:8000/predict/vision" \
     -F "file_foto=@contoh_apel.jpg"
```

---

## Out-of-Scope Detection

Model dilengkapi dua mekanisme untuk menolak gambar di luar dataset:

| Mekanisme           | Nilai default | Keterangan                                      |
|---------------------|---------------|-------------------------------------------------|
| Confidence threshold | 0.60         | Tolak jika skor tertinggi < threshold           |
| Entropy threshold    | 1.134        | Tolak jika distribusi probabilitas terlalu merata |

Nilai ini tersimpan di `model_metadata.json` dan dibaca otomatis saat startup.

---

## Training

Notebook pelatihan lengkap: [`vision_model.ipynb`](vision_model.ipynb)