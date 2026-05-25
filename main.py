from fastapi import FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import textwrap

# load env 
load_dotenv()

# download model
from scripts.download_models import download_models
download_models()

# import model
from model.utils import get_now
from model.schemas import ErrorResponseWrapper, ErrorDetail, MetaInfo
from model.vision.api_vision import router as vision_router
from model.genai.api_genai import router as genai_router
from model.regression.api_regression import router as regression_router

app = FastAPI(
    title="Wastefy AI Services",
    version="1.0.0",
    description=textwrap.dedent("""\
        Layanan AI untuk klasifikasi kondisi fisik sayur & buah dan panduan penyimpanannya.

        **Modul tersedia:**
        - **POST** `/predict/vision` — Klasifikasi kondisi fisik sayur & buah
        - **POST** `/predict/genai` — Panduan penyimpanan via Gemini AI

        **Format Response Standar:**
        ```json
        {
            "code": 200,
            "data": { ... },
            "message": "Deskripsi dalam Bahasa Indonesia",
            "meta": {
                "api": {"version": "1.0.0"},
                "generated_at": "ISO 8601",
                "model": {"name": "...", "version": "1.0.0"} | null
            },
            "status": "success | error"
        }
        ```
    """)
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router
app.include_router(vision_router)
app.include_router(genai_router)
app.include_router(regression_router)

# Root
@app.get("/")
async def root():
    return {
        "message": "API AI beroperasi dengan baik",
        "status": "aktif"
    }

# Health
@app.get("/health")
async def health():
    return {"status": "ok"}

# Handler 401
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return JSONResponse(status_code=401, content=ErrorResponseWrapper(
            code=status.HTTP_401_UNAUTHORIZED,
            errors=[ErrorDetail(error_code="unauthorized", message=str(exc.detail))],
            message="Akses ditolak",
            meta=MetaInfo(generated_at=get_now())
        ).model_dump())
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Handler 422
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = exc.errors()[0].get("msg", "Data tidak valid")
    return JSONResponse(status_code=422, content=ErrorResponseWrapper(
        code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        errors=[ErrorDetail(error_code="validation_error", message=error_msg)],
        message="Data tidak dapat diproses",
        meta=MetaInfo(generated_at=get_now())
    ).model_dump())