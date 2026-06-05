---
title: Wastefy AI Services
sdk: docker
pinned: false
---

# Wastefy AI Services

Entry point utama untuk seluruh layanan AI Wastefy berbasis FastAPI.

---

## Modul

| Modul | Prefix | Keterangan |
|---|---|---|
| Vision | `/predict/vision` | Klasifikasi kondisi fisik sayur & buah |
| Regression | `/predict/regression` | Prediksi sisa umur simpan |
| GenAI | `/predict/genai` | Panduan penyimpanan via Gemini AI |

---

## Tautan Model ML

File model Machine Learning (.keras) yang digunakan pada layanan ini di-hosting secara eksternal melalui Google Drive. 

Kamu dapat mengunduh model tersebut melalui tautan berikut:
- **[Link Folder/File Google Drive Model Wastefy](https://drive.google.com/drive/folders/103FMAtQpZtU_tKAtlYGadvcRrnmh2n9W?usp=sharing)**

*(Catatan: Akses untuk melihat dan mengunduh telah dibuka untuk akun `capstone@student.devacademy.id`)*

Setelah diunduh, letakkan file model tersebut di dalam direktori yang sesuai (misalnya di dalam folder `model/vision/` atau `model/regression/`) sebelum menjalankan server.

---

## Cara Pakai

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Atur Variabel Lingkungan

Salin file `.env.example` dan isi dengan API key milikmu:

```bash
# Linux/Mac
cp .env.example .env

# Windows
copy .env.example .env
```

Isi file `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
WASTEFY_API_KEY=your_secret_api_key_here
HF_TOKEN_RIIMARU=your_huggingface_token_riimaru_here
HF_TOKEN_ARCIII=your_huggingface_token_arciii_here
```

### 3. Jalankan Server

**Untuk Pengujian Lokal (Development):**

```bash
uvicorn main:app --reload
```

**Untuk Production (Hugging Face Spaces / Cloud):**

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 4. Akses Dokumentasi Swagger

- **Lokal:** `http://localhost:8000/docs`
- **Hugging Face:** `https://{username}-{space-name}.hf.space/docs`

---

## Struktur Project

```
project_root/
├── main.py                 ← Pusat kendali aplikasi
├── .env.example            ← Template variabel lingkungan
├── .gitignore
├── README.md
├── requirements.txt
│
└── model/                  ← Semua layanan AI ada di sini
    ├── __init__.py
    ├── config.py           ← Konfigurasi aplikasi
    ├── utils.py            ← Autentikasi & helper
    ├── schemas.py          ← Shared response schemas
    │
    ├── vision/             ← Klasifikasi kondisi fisik sayur & buah
    │   ├── __init__.py
    │   └── api_vision.py
    │
    ├── regression/         ← Prediksi sisa umur simpan
    │   ├── __init__.py
    │   └── api_regression.py
    │
    └── genai/              ← Panduan penyimpanan via Gemini AI
        ├── __init__.py
        ├── api_genai.py
        └── prompt_templates.json
```