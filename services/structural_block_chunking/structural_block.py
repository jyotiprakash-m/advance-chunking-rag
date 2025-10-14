import time
import logging
import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
)
from fastapi import UploadFile, HTTPException
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import from the correct module path - adjust based on your project structure
# from modals.request_response import StructuralBlockRequest, StructuralBlockResponse

# For now, let's define the models inline until the import path is fixed
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable, Literal
from modals.request_response import StructuralBlockRequest, StructuralBlockResponse


# Supported file formats
SUPPORTED_FORMATS = ['.pdf', '.txt', '.pptx', '.ppt', '.docx', '.doc', '.html', '.htm', '.xml']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 10MB limit

# Streaming and parallel processing constants
STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
MAX_PARALLEL_WORKERS = 4  # Maximum parallel processing workers
PROGRESS_UPDATE_INTERVAL = 0.1  # Progress update interval in seconds

# In-memory storage for progress (use Redis in production)
progress_store = {}

async def process_structural_chunking(request: StructuralBlockRequest, task_id: Optional[str] = None) -> StructuralBlockResponse:
        """
        Process structural block chunking with streaming and parallel processing.
        
        Args:
            request: StructuralBlockRequest containing all chunking parameters
            
        Returns:
            StructuralBlockResponse with chunking results
        """
        start_time = time.time()
        progress = {"current": 0, "total": 100, "message": "Starting processing"}
        
        def update_progress(step: int, message: str):
            progress["current"] = step
            progress["message"] = message
            logger.info(f"Progress: {step}% - {message}")
            if task_id:
                progress_store[task_id] = {"current": step, "total": 100, "message": message}
        
        try:
            update_progress(2, "Initializing processing")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)  # Allow progress to be visible
            
            update_progress(5, "Determining text source")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            # Determine text source: direct text or file
            if request.text:
                text_content = request.text
                filename = "direct_text.txt"
                file_extension = ".txt"
                logger.info("Using direct text input")
                update_progress(10, "Direct text input detected")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
                
            elif request.file:
                update_progress(8, "File input detected")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
                
                # Validate file format
                filename = request.file.filename or "unknown.txt"
                file_extension = get_file_extension(filename)
                if file_extension not in SUPPORTED_FORMATS:
                    logger.error(f"Unsupported file format: {file_extension}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                    )
                
                update_progress(12, "Validating file")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
                
                # Check file size
                file_size = len(await request.file.read())
                await request.file.seek(0)  # Reset file pointer
                if file_size > MAX_FILE_SIZE:
                    logger.error(f"File too large: {file_size} bytes")
                    raise HTTPException(
                        status_code=400,
                        detail=f"File size exceeds limit of {MAX_FILE_SIZE} bytes"
                    )
                
                update_progress(15, f"Reading file: {filename} ({file_size} bytes)")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
                
                # Stream and extract text from uploaded file with parallel processing
                logger.info(f"Streaming and extracting text from file: {filename}")
                text_content = await stream_extract_text_from_file(request.file, file_extension, task_id, update_progress)
                update_progress(45, "Text extraction completed")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
                
            else:
                logger.error("Neither file nor text provided")
                raise HTTPException(status_code=400, detail="Either file or text must be provided.")
            
            update_progress(50, "Configuring text splitter")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            logger.info(f"Text extracted, length: {len(text_content)} characters")
            # Configure text splitter based on request parameters
            text_splitter = configure_text_splitter(request)
            update_progress(55, "Splitter configured")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            update_progress(60, "Starting parallel text splitting")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            # Split text into chunks with parallel processing
            logger.info("Splitting text into chunks with parallel processing")
            chunks = await parallel_split_text(text_content, text_splitter, task_id, update_progress)
            update_progress(85, "Text splitting completed")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            # Convert to Document objects with metadata
            update_progress(90, "Creating document chunks")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, create_document_chunks, chunks, filename
            )
            
            # Apply chunk limit if specified
            if request.return_chunk_limit:
                chunks = chunks[:request.return_chunk_limit]
                logger.info(f"Limited to {len(chunks)} chunks")
            
            update_progress(95, "Preparing response")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            # Prepare response data with parallel processing
            chunk_data = await parallel_process_chunks(chunks, task_id, update_progress)
            
            processing_time = time.time() - start_time
            update_progress(100, "Processing complete")
            
            response = StructuralBlockResponse(
                status="success",
                message=f"Successfully processed {len(chunks)} structural blocks",
                total_chunks=len(chunks),
                chunks=chunk_data,
                processing_time=round(processing_time, 2),
                length_function_used=request.length_function or "characters",
                metadata={
                    "file_name": filename,
                    "file_type": file_extension,
                    "chunk_size": request.chunk_size,
                    "chunk_overlap": request.chunk_overlap,
                    "original_text_length": len(text_content),
                    "separators_used": request.separators or ["default"],
                    "language": request.language,
                    "section_headers": request.section_headers,
                    "progress": progress,
                    "processing_mode": "streaming_parallel"
                }
            )
            
            # Store the complete result in progress_store for WebSocket
            if task_id:
                progress_store[task_id] = {
                    "current": 100, 
                    "total": 100, 
                    "message": "Processing complete",
                    "result": response.dict()  # Convert to dict for JSON serialization
                }
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            progress["current"] = 0
            progress["message"] = f"Error: {str(e)}"
            logger.error(f"Error processing request: {str(e)}")
            
            error_response = StructuralBlockResponse(
                status="error",
                message=f"Error processing file: {str(e)}",
                total_chunks=0,
                chunks=[],
                processing_time=round(processing_time, 2),
                length_function_used=request.length_function or "characters",
                metadata={"error": str(e), "progress": progress}
            )
            
            # Store error result in progress_store
            if task_id:
                progress_store[task_id] = {
                    "current": 0, 
                    "total": 100, 
                    "message": f"Error: {str(e)}",
                    "result": error_response.dict()
                }
            
            return error_response
    
async def stream_extract_text_from_file(file: UploadFile, file_extension: str, task_id: Optional[str], update_progress) -> str:
    """Stream and extract text from file with progress updates."""
    try:
        # For text files, we can stream directly
        if file_extension == '.txt':
            content_chunks = []
            total_size = 0
            chunk_count = 0
            
            while True:
                chunk = await file.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                
                chunk_text = chunk.decode('utf-8')
                content_chunks.append(chunk_text)
                total_size += len(chunk)
                chunk_count += 1
                
                # Update progress during streaming
                progress_percent = min(15 + (chunk_count * 20) // 10, 40)  # Scale to 15-40%
                update_progress(progress_percent, f"Streaming text chunk {chunk_count}")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            return ''.join(content_chunks)
        
        # For binary files (PDF, DOCX, PPTX, etc.), read entire content first
        # then process in parallel if the extracted text is large
        else:
            update_progress(18, "Reading complete file for processing")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            content = await file.read()
            update_progress(25, f"Extracting text from {file_extension.upper()} file")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            # Extract text from the complete file
            text_content = await asyncio.get_event_loop().run_in_executor(
                None, extract_text_from_complete_file, content, file_extension
            )
            
            update_progress(40, f"Text extraction completed ({len(text_content)} characters)")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
            
            return text_content
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to stream extract text: {str(e)}")

def extract_text_from_complete_file(content: bytes, file_extension: str) -> str:
    """Extract text from complete file content."""
    try:
        if file_extension == '.pdf':
            return extract_text_from_pdf(content)
        elif file_extension in ['.pptx', '.ppt']:
            return extract_text_from_ppt(content)
        elif file_extension in ['.docx', '.doc']:
            return extract_text_from_doc(content)
        elif file_extension in ['.html', '.htm']:
            return extract_text_from_html(content)
        elif file_extension == '.xml':
            return extract_text_from_xml(content)
        else:
            return content.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.warning(f"Error extracting text from {file_extension}: {str(e)}")
        return ""

async def parallel_split_text(text_content: str, text_splitter, task_id: Optional[str], update_progress) -> List[str]:
    """Split text into chunks using parallel processing."""
    # For large texts, split into segments and process in parallel
    if len(text_content) > 50000:  # 50KB threshold for parallel processing
        segment_size = max(10000, len(text_content) // MAX_PARALLEL_WORKERS)  # Minimum 10KB segments
        text_segments = []
        
        for i in range(0, len(text_content), segment_size):
            segment = text_content[i:i + segment_size]
            # Ensure we don't break in the middle of a word/sentence
            if i + segment_size < len(text_content):
                # Find a good break point (sentence end or word boundary)
                break_chars = ['. ', '! ', '? ', '\n\n', '\n']
                break_pos = -1
                for char in break_chars:
                    pos = segment.rfind(char)
                    if pos > len(segment) * 0.7:  # Break in last 30% of segment
                        break_pos = max(break_pos, pos + len(char))
                
                if break_pos > 0:
                    segment = segment[:break_pos]
                    # Adjust next segment start
                    i = i + break_pos - len(char) if 'char' in locals() else i + break_pos
            
            text_segments.append(segment)
        
        update_progress(65, f"Split text into {len(text_segments)} segments for parallel processing")
        await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
        
        # Process segments in parallel
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(text_segments))) as executor:
            tasks = []
            for i, segment in enumerate(text_segments):
                task = loop.run_in_executor(executor, text_splitter.split_text, segment)
                tasks.append(task)
            
            # Collect results with progress updates
            segment_results = []
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                segment_results.extend(result)
                completed += 1
                
                # Update progress
                progress_percent = 65 + int((completed / len(tasks)) * 15)  # Scale to 65-80%
                update_progress(progress_percent, f"Split segment {completed}/{len(tasks)}")
                await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
        
        return segment_results
    else:
        # For smaller texts, split directly
        update_progress(70, "Splitting text directly")
        await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
        
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, text_splitter.split_text, text_content)
        
        update_progress(80, "Text splitting completed")
        await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
        
        return chunks

async def parallel_process_chunks(chunks: List[Document], task_id: Optional[str], update_progress) -> List[Dict]:
    """Process chunks in parallel to create response data."""
    update_progress(90, f"Processing {len(chunks)} chunks in parallel")
    await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
    
    # Split chunks into batches for parallel processing
    batch_size = max(1, len(chunks) // MAX_PARALLEL_WORKERS)
    chunk_batches = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        chunk_batches.append(batch)
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        tasks = []
        for batch in chunk_batches:
            task = loop.run_in_executor(executor, process_chunk_batch, batch)
            tasks.append(task)
        
        # Collect results
        batch_results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            batch_results.extend(result)
            completed += 1
            
            # Update progress
            progress_percent = 90 + int((completed / len(tasks)) * 8)  # Scale to 90-98%
            update_progress(progress_percent, f"Processed batch {completed}/{len(tasks)}")
            await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)
    
    # Sort by original index
    batch_results.sort(key=lambda x: x['index'])
    return batch_results

def process_chunk_batch(chunk_batch: List[Document]) -> List[Dict]:
    """Process a batch of chunks to create response data."""
    results = []
    enc = tiktoken.get_encoding("cl100k_base")
    
    for chunk in chunk_batch:
        chunk_data = {
            "index": chunk.metadata.get("chunk_index", 0),
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "word_count": len(chunk.page_content.split()),
            "char_count": len(chunk.page_content),
            "token_count": len(enc.encode(chunk.page_content))
        }
        results.append(chunk_data)
    
    return results
    
def get_file_extension(filename: str) -> str:
    """Extract file extension from filename."""
    if not filename:
        raise ValueError("Filename is required")
    return os.path.splitext(filename)[1].lower()

async def extract_text_from_file(file: UploadFile, file_extension: str) -> str:
    """Extract text content from uploaded file."""
    try:
        content = await file.read()

        if file_extension == '.pdf':
            return extract_text_from_pdf(content)
        elif file_extension in ['.pptx', '.ppt']:
            return extract_text_from_ppt(content)
        elif file_extension in ['.docx', '.doc']:
            return extract_text_from_doc(content)
        elif file_extension in ['.html', '.htm']:
            return extract_text_from_html(content)
        elif file_extension == '.xml':
            return extract_text_from_xml(content)
        elif file_extension == '.txt':
            return content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content using LangChain PyPDFLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_content)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents if doc.page_content)
    finally:
        os.unlink(temp_path)


def extract_text_from_ppt(ppt_content: bytes) -> str:
    """Extract text from PowerPoint content using LangChain UnstructuredPowerPointLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as temp_file:
        temp_file.write(ppt_content)
        temp_path = temp_file.name

    try:
        loader = UnstructuredPowerPointLoader(temp_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents if doc.page_content)
    finally:
        os.unlink(temp_path)


def extract_text_from_doc(doc_content: bytes) -> str:
    """Extract text from Word document content using LangChain UnstructuredWordDocumentLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
        temp_file.write(doc_content)
        temp_path = temp_file.name

    try:
        loader = UnstructuredWordDocumentLoader(temp_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents if doc.page_content)
    finally:
        os.unlink(temp_path)


def extract_text_from_html(html_content: bytes) -> str:
    """Extract text from HTML content using LangChain UnstructuredHTMLLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_path = temp_file.name

    try:
        loader = UnstructuredHTMLLoader(temp_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents if doc.page_content)
    finally:
        os.unlink(temp_path)


def extract_text_from_xml(xml_content: bytes) -> str:
    """Extract text from XML content using LangChain UnstructuredXMLLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as temp_file:
        temp_file.write(xml_content)
        temp_path = temp_file.name

    try:
        loader = UnstructuredXMLLoader(temp_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents if doc.page_content)
    finally:
        os.unlink(temp_path)

def get_length_function(length_type: str) -> Callable[[str], int]:
    """
    Get the appropriate length function based on type.
    
    Args:
        length_type: Type of length calculation:
            - "characters": Count actual characters (default)
            - "words": Count words (space-separated tokens)
            - "tokens": Accurate token count using tiktoken (GPT-3.5/4 tokenizer)
    
    Returns:
        Function that takes a string and returns its length as an integer
    """
    if length_type == "words":
        return lambda text: len(text.split())
    elif length_type == "tokens":
        # Use tiktoken for accurate token counting
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
        return lambda text: len(enc.encode(text))
    else:  # default to "characters"
        return len

def configure_text_splitter(request: StructuralBlockRequest):
    """Configure the text/code splitter based on request parameters."""
    # If language is a supported programming language, use RecursiveCharacterTextSplitter.from_language
    if request.language and request.language.lower() in [lang.value for lang in Language]:
        lang_enum = Language(request.language.lower())
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    # Otherwise, use RecursiveCharacterTextSplitter for text
    kwargs: Dict[str, Any] = {
        'chunk_size': request.chunk_size,
        'chunk_overlap': request.chunk_overlap,
    }
    if request.separators is not None:
        kwargs['separators'] = request.separators
    if request.length_function is not None:
        kwargs['length_function'] = get_length_function(request.length_function)
    if request.keep_separator is not None:
        kwargs['keep_separator'] = request.keep_separator
    return RecursiveCharacterTextSplitter(**kwargs)

def create_document_chunks(text_chunks: List[str], filename: str) -> List[Document]:
    """Create Document objects with metadata from text chunks."""
    documents = []
    
    for idx, chunk in enumerate(text_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_index": idx,
                "source": filename,
                "chunk_type": "structural_block",
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            }
        )
        documents.append(doc)
    
    return documents
