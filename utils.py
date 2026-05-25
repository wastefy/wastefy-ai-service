import os
import json
import tensorflow as tf
from datetime import datetime, timezone
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from model.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def load_model(model_type: str):
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, model_type, "model.keras")
    METADATA_PATH = os.path.join(BASE_DIR, model_type, "model_metadata.json")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"File metadata tidak ditemukan: {METADATA_PATH}")

    # Load model & metadata
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata

def get_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

async def verify_api_key(api_key_str: str = Security(api_key_header)):
    if api_key_str != settings.WASTEFY_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key tidak valid"
        )
    return api_key_str