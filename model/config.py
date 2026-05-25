from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    WASTEFY_API_KEY: str

    HF_TOKEN_RIIMARU: Optional[str] = None
    HF_TOKEN_ARCIII: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()