# Wastefy AI Services

Entry point utama untuk seluruh layanan AI Wastefy berbasis FastAPI.

## Modul

| Modul | Prefix | Keterangan |
|---|---|---|
| Vision | `/predict/vision` | Deteksi visual bahan baku |
| Regression | `/predict/regression` | Prediksi sisa umur simpan |
| GenAI | `/predict/genai` | Panduan penyimpanan via Gemini AI |

## Menjalankan Server

```bash
uvicorn main:app --reload
```

Dokumentasi Swagger tersedia di `http://localhost:8000/docs`.
