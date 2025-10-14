from pydantic import BaseModel, model_validator
# from typing import List, Optional, Literal, Any
from fastapi import UploadFile

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class StructuralBlockRequest(BaseModel):
    file: Optional[UploadFile] = None
    text: Optional[str] = None
    chunk_size: int
    chunk_overlap: int
    separators: Optional[List[str]] = None
    length_function: Optional[Literal["characters", "words", "tokens"]] = None
    keep_separator: Optional[bool] = None
    visualize_count: Optional[int] = None
    section_headers: Optional[List[str]] = None
    language: Optional[str] = None
    return_chunk_limit: Optional[int] = None

    @model_validator(mode="after")
    def check_file_or_text(self):
        if not self.file and not self.text:
            raise ValueError('Either file or text must be provided.')
        return self

class StructuralBlockResponse(BaseModel):
    """Response model for structural block chunking."""
    status: str
    message: str
    chunks: List[Dict[str, Any]]
    total_chunks: int
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    length_function_used: Optional[str] = None