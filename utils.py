from datetime import datetime, timezone
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from model.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

async def verify_api_key(api_key_str: str = Security(api_key_header)):
    if api_key_str != settings.WASTEFY_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key tidak valid"
        )
    return api_key_str