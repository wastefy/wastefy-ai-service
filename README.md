# GENAI (Panduan Penyimpanan Bahan Baku via Gemini AI)

Modul FastAPI untuk menghasilkan panduan penyimpanan bahan baku dapur katering secara otomatis menggunakan Google Gemini AI.

---

## Struktur Folder

```
feat/genai/
├── .env.example
├── .gitignore
├── api_genai.py
├── prompt_templates.txt
├── README.md
└── requirements.txt
```

---

## Prasyarat

- Python 3.9+
- API Key dari [Google AI Studio](https://aistudio.google.com/)

---

## Instalasi

1. Clone repository dan masuk ke direktori project.

2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Salin file environment dan isi API key:
   ```bash
   cp .env.example .env
   ```
   Kemudian edit `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

---

## Menjalankan Server

```bash
uvicorn main:app --reload
```

---

## Endpoint

### `POST /predict/genai`

Menghasilkan panduan penyimpanan untuk satu bahan baku.

**Request Body:**

```json
{
  "nama_item": "Ayam Fillet",
  "jenis_item": "Protein Hewani",
  "kondisi_fisik": "Segar",
  "lokasi_penyimpanan": "Chiller",
  "sisa_hari": 2
}
```

| Field | Tipe | Keterangan |
|---|---|---|
| `nama_item` | string | Nama bahan baku |
| `jenis_item` | string | Kategori bahan (Sayur, Buah) |
| `kondisi_fisik` | string | Kondisi fisik saat ini (Mentah, Matang, Terlalu Matang, Segar, Busuk) |
| `lokasi_penyimpanan` | string | Lokasi simpan saat ini (misal: Suhu Ruang, Pendingin, Pembeku) |
| `sisa_hari` | integer | Perkiraan sisa umur simpan dalam hari |

**Response Sukses (`200 OK`):**

```json
{
  "status": "success",
  "teks_panduan": "- Tindakan Prioritas: Gunakan hari ini sebelum kualitas menurun\n- Cara Simpan: Simpan di chiller suhu 0–4°C dalam wadah tertutup\n- Tips Katering: Pisahkan dari bahan matang untuk cegah kontaminasi"
}
```

**Response Error:**

```json
{
  "error": "pesan error"
}
```

---

## Catatan

- File `.env` **tidak boleh** di-commit ke repository. Pastikan `.gitignore` sudah mencantumkan `.env`.
- Model yang digunakan: `gemini-3.5-flash` (dapat diubah di `api_genai.py`).
- Output AI dibatasi dengan temperature 0.2 untuk hasil yang konsisten.