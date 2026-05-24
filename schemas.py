from pydantic import BaseModel
from typing import Optional, List, TypeVar, Generic

T = TypeVar("T")

# OUTPUT SCHEMAS

class MetaInfo(BaseModel):
    api: dict = {"version": "1.0.0"}
    generated_at: str
    model: Optional[dict] = None

class SuccessResponse(BaseModel, Generic[T]):
    code: int = 200
    data: T
    message: str
    meta: MetaInfo
    status: str = "success"

class ErrorDetail(BaseModel):
    error_code: str
    message: str

class ErrorResponseWrapper(BaseModel):
    code: int
    errors: List[ErrorDetail]
    message: str
    meta: MetaInfo
    status: str = "error"