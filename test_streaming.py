#!/usr/bin/env python3
"""
Test script for streaming and parallel processing chunking
"""
import asyncio
import aiohttp
import json
import time

async def test_chunking():
    """Test the chunking API with streaming and parallel processing"""

    # Test data - larger text to trigger parallel processing
    test_text = """
    This is a comprehensive test document for streaming and parallel processing.
    It contains multiple paragraphs to test the chunking functionality with realistic content.

    Introduction Section:
    This document demonstrates the advanced text processing capabilities of our system.
    The implementation includes streaming file processing, parallel text chunking, and real-time progress updates.
    These features provide significant performance improvements for large documents.

    Technical Implementation:
    The system uses asyncio for asynchronous processing and ThreadPoolExecutor for parallel operations.
    Text files are streamed in chunks to avoid memory issues with large files.
    Binary files like PDFs are processed completely before parallel text splitting occurs.
    Progress updates are sent frequently to provide realistic WebSocket feedback.

    Performance Benefits:
    - Faster processing through parallel execution of independent tasks
    - Memory-efficient streaming for large text files
    - Real-time progress updates every few seconds
    - Better user experience with granular feedback
    - Scalable architecture that handles various file types

    Advanced Features:
    The chunking algorithm supports multiple splitting strategies including character-based,
    word-based, and token-based splitting. Language-aware splitting is available for
    programming languages with proper syntax preservation.

    Quality Assurance:
    Each chunk maintains metadata including word count, character count, and token count.
    The system validates file types and sizes before processing.
    Error handling ensures graceful failure with detailed error messages.
    """ * 50  # Repeat to create a large document (about 50KB)

    # API endpoint
    url = "http://localhost:8000/api/v1/operations/structural-block"

    # Request data
    data = {
        "text": test_text,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "length_function": "characters"
    }

    print("Starting chunking test with streaming and parallel processing...")
    print(f"Text length: {len(test_text)} characters")

    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time

                    print("\n✅ Chunking completed successfully!")
                    print(f"Processing time: {processing_time:.2f} seconds")
                    print(f"Total chunks: {result['total_chunks']}")
                    print(f"Processing mode: {result['metadata'].get('processing_mode', 'unknown')}")

                    # Show first few chunks
                    print("\nFirst 3 chunks:")
                    for i, chunk in enumerate(result['chunks'][:3]):
                        print(f"Chunk {i+1}: {len(chunk['content'])} chars, {chunk['word_count']} words")

                else:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status} - {error_text}")

    except Exception as e:
        print(f"❌ Connection error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_chunking())