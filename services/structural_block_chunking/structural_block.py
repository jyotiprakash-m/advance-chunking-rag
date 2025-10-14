import time
import logging
import tiktoken
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

# In-memory storage for progress (use Redis in production)
progress_store = {}

async def process_structural_chunking(request: StructuralBlockRequest, task_id: Optional[str] = None) -> StructuralBlockResponse:
        """
        Process structural block chunking based on the request parameters.
        
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
            update_progress(5, "Initializing processing")
            update_progress(10, "Determining text source")
            # Determine text source: direct text or file
            if request.text:
                text_content = request.text
                filename = "direct_text.txt"
                file_extension = ".txt"
                logger.info("Using direct text input")
                update_progress(15, "Direct text input detected")
            elif request.file:
                update_progress(15, "File input detected")
                # Validate file format
                filename = request.file.filename or "unknown.txt"
                file_extension = get_file_extension(filename)
                if file_extension not in SUPPORTED_FORMATS:
                    logger.error(f"Unsupported file format: {file_extension}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                    )
                
                update_progress(20, "Validating file")
                # Check file size
                file_size = len(await request.file.read())
                await request.file.seek(0)  # Reset file pointer
                if file_size > MAX_FILE_SIZE:
                    logger.error(f"File too large: {file_size} bytes")
                    raise HTTPException(
                        status_code=400,
                        detail=f"File size exceeds limit of {MAX_FILE_SIZE} bytes"
                    )
                
                update_progress(25, f"Reading file: {filename}")
                # Extract text from uploaded file
                logger.info(f"Extracting text from file: {filename}")
                text_content = await extract_text_from_file(request.file, file_extension)
                update_progress(40, "Text extraction completed")
            else:
                logger.error("Neither file nor text provided")
                raise HTTPException(status_code=400, detail="Either file or text must be provided.")
            
            update_progress(50, "Configuring text splitter")
            logger.info(f"Text extracted, length: {len(text_content)} characters")
            # Configure text splitter based on request parameters
            text_splitter = configure_text_splitter(request)
            update_progress(60, "Splitter configured")
            
            update_progress(70, "Starting text splitting")
            # Split text into chunks
            logger.info("Splitting text into chunks")
            documents = text_splitter.split_text(text_content)
            update_progress(80, "Text splitting completed")
            
            # Convert to Document objects with metadata
            chunks = create_document_chunks(documents, filename)
            
            # Apply chunk limit if specified
            if request.return_chunk_limit:
                chunks = chunks[:request.return_chunk_limit]
                logger.info(f"Limited to {len(chunks)} chunks")
            
            update_progress(90, "Preparing response")
            logger.info(f"Successfully processed {len(chunks)} chunks")
            # Prepare response data
            chunk_data = [
                {
                    "index": idx,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "word_count": len(chunk.page_content.split()),
                    "char_count": len(chunk.page_content),
                    "token_count": len(tiktoken.get_encoding("cl100k_base").encode(chunk.page_content))
                }
                for idx, chunk in enumerate(chunks)
            ]
            
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
                    "progress": progress
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
