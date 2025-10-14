from typing import List, Optional, Literal
from fastapi import APIRouter, UploadFile, File, Query

from services.structural_block_chunking.structural_block import process_structural_chunking
from modals.request_response import StructuralBlockRequest, StructuralBlockResponse

router = APIRouter(prefix="/operations", tags=["Operations"])

@router.post(
    "/structural-block",
    response_model=StructuralBlockResponse,
    summary="Chunk documents using structural block strategy",
)
async def structural_block_chunking_api(
    file: Optional[UploadFile] = File(None, description="Input document (.pdf, .txt, etc.) - optional if text is provided"),
    text: Optional[str] = Query(None, description="Direct text input - optional if file is provided"),
    chunk_size: int = Query(1000, ge=1, description="Max size per chunk"),
    chunk_overlap: int = Query(200, ge=0, description="Overlap between chunks"),
    separators: Optional[List[str]] = Query(None, description="Custom separators"),
    length_function: Optional[Literal["characters", "words", "tokens"]] = Query(
        None, description="Length metric to use"
    ),
    keep_separator: Optional[bool] = Query(None, description="Keep separators in chunks"),
    visualize_count: Optional[int] = Query(None, ge=1, description="Preview limit for chunks"),
    section_headers: Optional[List[str]] = Query(None, description="Section headers to prioritize"),
    language: Optional[Literal["python", "javascript", "typescript", "java", "cpp", "c", "csharp", "go", "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "shell", "sql", "html", "css", "xml", "json", "yaml"]] = Query(None, description="Programming language for code-aware chunking (optional)"),
    return_chunk_limit: Optional[int] = Query(None, ge=1, description="Limit number of chunks returned"),
):
    """Perform structural block chunking on a document or text."""
    
    request = StructuralBlockRequest(
        file=file,
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=length_function,
        keep_separator=keep_separator,
        visualize_count=visualize_count,
        section_headers=section_headers,
        language=language,
        return_chunk_limit=return_chunk_limit,
    )

    return await process_structural_chunking(request)
