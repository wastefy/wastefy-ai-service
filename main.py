from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import semua router langsung dari subfoldernya masing-masing
from model.vision.api_vision import router as api_vision
from model.regression.api_regression import router as api_regression
from model.genai.api_genai import router as api_genai

app = FastAPI(title="Wastefy AI Services")

# Konfigurasi keamanan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menjahit semua modul ke dalam aplikasi utama
app.include_router(api_vision)
app.include_router(api_regression)
app.include_router(api_genai)

@app.get("/")
async def root():
    return {
        "message": "API AI beroperasi dengan baik",
        "status": "aktif"
    }