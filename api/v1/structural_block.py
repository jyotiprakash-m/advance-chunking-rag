from typing import List, Optional, Literal
import asyncio
import uuid
from fastapi import APIRouter, UploadFile, File, Query, WebSocket, BackgroundTasks

from services.structural_block_chunking.structural_block import process_structural_chunking, progress_store
from modals.request_response import StructuralBlockRequest, StructuralBlockResponse, TaskInitiationResponse

router = APIRouter(prefix="/operations", tags=["Operations"])

async def process_with_progress(request: StructuralBlockRequest, task_id: str):
    """Process chunking with progress updates."""
    try:
        # Call the service function (which now updates progress internally)
        result = await process_structural_chunking(request, task_id)
        progress_store[task_id] = {"current": 100, "total": 100, "message": "Completed", "result": result.model_dump()}
    except Exception as e:
        progress_store[task_id] = {"current": 0, "total": 100, "message": f"Error: {str(e)}", "error": str(e)}

@router.websocket("/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            if task_id in progress_store:
                progress = progress_store[task_id]
                await websocket.send_json(progress)
                if progress["current"] >= 100 or progress.get("error"):
                    break
            await asyncio.sleep(0.5)  # Poll every 0.5 seconds
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@router.post(
    "/structural-block",
    response_model=TaskInitiationResponse,
    summary="Chunk documents using structural block strategy",
)
async def structural_block_chunking_api(
    background_tasks: BackgroundTasks,
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
    
    task_id = str(uuid.uuid4())  # Generate unique task ID
    progress_store[task_id] = {"current": 0, "total": 100, "message": "Task queued"}
    
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

    # Run processing in background
    background_tasks.add_task(process_with_progress, request, task_id)
    
    return TaskInitiationResponse(
        task_id=task_id,
        status="processing",
        message="Processing started. Use WebSocket at /operations/progress/{task_id} for real-time updates."
    )
