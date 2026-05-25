# GenAI (Panduan Penyimpanan Bahan Baku via Gemini AI)

Modul FastAPI untuk menghasilkan panduan penyimpanan sayur dan buah secara otomatis menggunakan Google Gemini AI.

---

## Struktur Folder

```
model/genai/
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

## Cara Pakai

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Salin file environment dan isi API key:
   ```bash
   cp .env.example .env
   ```
   Kemudian edit `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   WASTEFY_API_KEY=your_secret_api_key_here
   ```

3. Daftarkan router ke `main.py`:
    ```python
    from model.genai.api_genai import router as genai_router

    app.include_router(genai_router)
    ```

4. Jalankan Server

    ```bash
    uvicorn main:app --reload
    ```

5. Uji Endpoint via cURL

    ```bash
    curl -X POST "http://localhost:8000/predict/genai" \
        -H "X-API-Key: kunci_rahasia_123" \
        -H "Content-Type: application/json" \
        -d '{
          "nama_item": "Apel",
          "jenis_item": "Buah",
          "kondisi_fisik": "Matang",
          "lokasi_penyimpanan": "Pendingin",
          "sisa_hari": 5
        }'
    ```
---

## Endpoint

### `POST /predict/genai`

Menghasilkan panduan penyimpanan untuk satu bahan baku.

#### Request Headers

```http
X-API-Key: <kunci_rahasia_api_anda>
Content-Type: application/json
```

#### Request Body

```json
{
  "nama_item": "Apel",
  "jenis_item": "Buah",
  "kondisi_fisik": "Matang",
  "lokasi_penyimpanan": "Pendingin",
  "sisa_hari": 5
}
```

| Field | Tipe | Nilai Valid |
|---|---|---|
| `nama_item` | string | Anggur, Apel, Cabai, Jeruk, Kentang, Mangga, Mentimun, Pisang, Tomat, Wortel |
| `jenis_item` | string | Buah, Sayur |
| `kondisi_fisik` | string | Busuk, Matang, Mentah, Terlalu Matang, Segar |
| `lokasi_penyimpanan` | string | Suhu Ruang, Pendingin, Pembeku |
| `sisa_hari` | integer | 0–40 hari |

#### Respons Sukses (`200 OK`)

```json
{
  "status": "success"
  "code": 200,
  "data": {
    "cara_simpan": "- Tindakan Prioritas: Segera pindah ke tempat sejuk...\n- Cara Simpan: Simpan dalam wadah tertutup rapat...\n- Tips Tambahan: Pisahkan dari bahan berbau kuat..."
  },
  "message": "Panduan berhasil dibuat",
  "meta": {
    "api": { "version": "1.0.0" },
    "generated_at": "2026-05-24T02:00:00Z",
    "model": null
  },
}
```

#### Respons Error

| Kode | Keterangan |
|---|---|
| `401` | API Key tidak valid atau tidak ada |
| `422` | Input tidak sesuai nilai valid |
| `500` | Gagal terhubung ke Gemini API |

---

## Catatan

- File `.env` **tidak boleh** di-commit ke repository. Pastikan `.gitignore` sudah mencantumkan `.env`.
- Output AI dibatasi dengan `temperature=0.2` untuk hasil yang konsisten.
- Jika `sisa_hari` bernilai `0`, kondisi otomatis dikirim sebagai `Kedaluwarsa/Perlu Segera Diolah` ke prompt.