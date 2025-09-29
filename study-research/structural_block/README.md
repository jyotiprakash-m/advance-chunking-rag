# PDF Structural Block Chunking Demo

This project demonstrates advanced PDF chunking for RAG (Retrieval-Augmented Generation) applications. It extracts, analyzes, and stores rich, hierarchical information from PDF documents, enabling powerful semantic and structural search.

## Key Features
- **Font Size-Based Heading Detection:** Automatically identifies headings and sections using font size and layout.
- **Hierarchical Chunking:** Chunks are linked with parent-child relationships, preserving document structure.
- **Rich Metadata:** Each chunk stores text, font info, page number, position, hierarchy level, parent/child IDs, and more.
- **Semantic Search:** Chunks are stored in ChromaDB with embeddings for context-aware retrieval.
- **Q&A Demo:** Answers user questions by searching through structured chunks for relevant information.

## Example Chunk Storage
```python
Document(
    page_content="Full Stack Developer at SettleMint India, Delhi (January 2023 - Present)",
    metadata={
        "source_file": "study-research/Resume JPM.pdf",
        "page_number": 1,
        "font_size": 11.0,
        "font_name": "Arial-BoldMT",
        "x0": 72.0, "y0": 150.0,
        "x1": 540.0, "y1": 170.0,
        "is_heading": False,
        "chunk_id": "chunk_0005",
        "chunk_type": "pdf_content",
        "hierarchy_level": 2,
        "parent_id": "chunk_0002",
        "child_ids": [],
        "word_count": 10,
        "content_length": 68
    }
)
```

## Why So Much Metadata?
- Enables semantic and hierarchical search
- Preserves context for Q&A and document understanding
- Supports filtering by section, heading, or content

## Parallel Chunking
You can use threading (e.g., `ThreadPoolExecutor`) to process large PDFs faster by chunking pages in parallel.

## Usage
Run `structural-block-pdf.py` to see chunking, storage, and Q&A in action.

---

**This system is ideal for RAG, document analysis, and any application needing deep, context-aware PDF understanding.**
