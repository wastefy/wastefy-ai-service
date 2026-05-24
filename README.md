# Wastefy AI Services

Entry point utama untuk seluruh layanan AI Wastefy berbasis FastAPI.

---

## Modul

| Modul | Prefix | Keterangan |
|---|---|---|
| Vision | `/predict/vision` | Klasifikasi kondisi fisik sayur & buah |
| GenAI | `/predict/genai` | Panduan penyimpanan via Gemini AI |

> Modul Regression belum tersedia.

---

## Menjalankan Server

**Untuk Pengujian Lokal (Development):**
```bash
uvicorn main:app --reload
```

**Untuk Production (Hugging Face Spaces / Cloud):**
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Dokumentasi Swagger tersedia di:
- **Lokal:** `http://localhost:8000/docs`
- **Hugging Face:** `https://{username}-{space-name}.hf.space/docs`