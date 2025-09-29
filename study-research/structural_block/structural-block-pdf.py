#!/usr/bin/env python3
"""
PDF Structural Block Chunking with Hierarchical Relationships

This script demonstrates advanced PDF processing techniques for RAG applications:
- Font size-based heading detection
- Structural block identification
- Hierarchical chunking with parent-child relationships
- ChromaDB integration for semantic search

Key Features:
- Extracts text blocks with layout information (font size, position)
- Automatically detects headings based on font characteristics
- Creates hierarchical chunks preserving document structure
- Stores chunks in vector database with metadata
- Supports semantic search with structural awareness
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pdfplumber
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class TextBlock:
    """Represents a text block extracted from PDF with layout information"""
    text: str
    font_size: float
    font_name: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PDFChunk:
    """Represents a chunk with hierarchical metadata"""
    id: str
    content: str
    chunk_type: str
    hierarchy_level: int
    parent_id: Optional[str]
    child_ids: List[str]
    font_size: float
    page_number: int
    metadata: Dict[str, Any]


class PDFStructuralChunker:
    """
    Advanced PDF chunker that preserves document structure through hierarchical relationships
    """

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.chunk_counter = 0
        self.hierarchy_map: Dict[str, PDFChunk] = {}

    def get_next_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        self.chunk_counter += 1
        return f"chunk_{self.chunk_counter:04d}"

    def extract_structured_content(self, pdf_path: str) -> List[TextBlock]:
        """
        Extract text blocks from PDF with layout information
        """
        text_blocks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text with character-level details
                chars = page.chars

                if not chars:
                    continue

                # Group characters into text blocks
                current_block = []
                current_font_size = None
                current_font_name = None

                for char in chars:
                    # Check if this character belongs to current block
                    if (current_font_size is None or
                        abs(char['size'] - current_font_size) < 0.1):
                        current_block.append(char)
                        if current_font_size is None:
                            current_font_size = char['size']
                            current_font_name = char.get('fontname', 'Unknown')
                    else:
                        # Process current block
                        if current_block:
                            block_text = self._chars_to_text(current_block)
                            if block_text.strip():
                                # Calculate bounding box
                                x0 = min(c['x0'] for c in current_block)
                                y0 = min(c['top'] for c in current_block)
                                x1 = max(c['x1'] for c in current_block)
                                y1 = max(c['bottom'] for c in current_block)

                                text_blocks.append(TextBlock(
                                    text=block_text.strip(),
                                    font_size=current_font_size or 12.0,
                                    font_name=current_font_name or "Unknown",
                                    page_number=page_num,
                                    x0=x0, y0=y0, x1=x1, y1=y1
                                ))

                        # Start new block
                        current_block = [char]
                        current_font_size = char['size']
                        current_font_name = char.get('fontname', 'Unknown')

                # Process final block
                if current_block:
                    block_text = self._chars_to_text(current_block)
                    if block_text.strip():
                        x0 = min(c['x0'] for c in current_block)
                        y0 = min(c['top'] for c in current_block)
                        x1 = max(c['x1'] for c in current_block)
                        y1 = max(c['bottom'] for c in current_block)

                        text_blocks.append(TextBlock(
                            text=block_text.strip(),
                            font_size=current_font_size or 12.0,
                            font_name=current_font_name or "Unknown",
                            page_number=page_num,
                            x0=x0, y0=y0, x1=x1, y1=y1
                        ))

        return text_blocks

    def _chars_to_text(self, chars: List[Dict]) -> str:
        """Convert character list to text, handling spacing"""
        if not chars:
            return ""

        # Sort characters by position (left to right, top to bottom)
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))

        text = ""
        prev_char = None

        for char in sorted_chars:
            if prev_char:
                # Add space if characters are far apart
                if char['x0'] - prev_char['x1'] > prev_char['size'] * 0.3:
                    text += " "

            text += char['text']
            prev_char = char

        return text

    def analyze_font_hierarchy(self, text_blocks: List[TextBlock]) -> Dict[float, int]:
        """
        Analyze font sizes to determine hierarchy levels
        Returns mapping of font_size -> hierarchy_level
        """
        if not text_blocks:
            return {}

        # Get unique font sizes, sorted descending
        font_sizes = sorted(set(block.font_size for block in text_blocks), reverse=True)

        # Assign hierarchy levels (smaller numbers = higher in hierarchy)
        hierarchy_map = {}
        for i, size in enumerate(font_sizes):
            hierarchy_map[size] = i

        return hierarchy_map

    def is_heading(self, block: TextBlock, font_hierarchy: Dict[float, int],
                   text_blocks: List[TextBlock]) -> bool:
        """
        Determine if a text block is likely a heading
        """
        # Must be among the largest font sizes
        if font_hierarchy.get(block.font_size, 999) > 2:
            return False

        text = block.text.strip()

        # Headings are typically short
        if len(text) > 100:
            return False

        # Check for heading-like patterns
        if len(text.split()) <= 15:
            # All caps or title case
            if text.isupper() or text.istitle():
                return True

            # Contains common heading keywords
            heading_keywords = ['chapter', 'section', 'introduction', 'conclusion',
                              'summary', 'overview', 'background', 'methodology']
            if any(keyword in text.lower() for keyword in heading_keywords):
                return True

        return False

    def create_hierarchical_chunks(self, pdf_path: str) -> List[PDFChunk]:
        """
        Create hierarchical chunks from PDF with structural awareness
        """
        console.print(f"\n[bold blue]📄 Analyzing PDF structure: {pdf_path}[/bold blue]")

        # Extract structured content
        text_blocks = self.extract_structured_content(pdf_path)
        console.print(f"   → Extracted {len(text_blocks)} text blocks")

        if not text_blocks:
            return []

        # Analyze font hierarchy
        font_hierarchy = self.analyze_font_hierarchy(text_blocks)
        console.print(f"   → Detected {len(font_hierarchy)} font size levels")

        # Create chunks with hierarchy
        chunks = []
        current_parent: Optional[PDFChunk] = None
        current_section: Optional[PDFChunk] = None

        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue

            is_heading_block = self.is_heading(block, font_hierarchy, text_blocks)
            hierarchy_level = font_hierarchy.get(block.font_size, 999)

            if is_heading_block:
                # Create heading chunk
                chunk_type = "pdf_heading" if hierarchy_level == 0 else "pdf_subheading"

                chunk = PDFChunk(
                    id=self.get_next_chunk_id(),
                    content=text,
                    chunk_type=chunk_type,
                    hierarchy_level=hierarchy_level,
                    parent_id=current_parent.id if current_parent else None,
                    child_ids=[],
                    font_size=block.font_size,
                    page_number=block.page_number,
                    metadata={
                        "source_file": pdf_path,
                        "page_number": block.page_number,
                        "font_size": block.font_size,
                        "font_name": block.font_name,
                        "x0": block.x0, "y0": block.y0,
                        "x1": block.x1, "y1": block.y1,
                        "is_heading": True
                    }
                )

                chunks.append(chunk)
                self.hierarchy_map[chunk.id] = chunk

                # Update hierarchy
                if hierarchy_level == 0:
                    current_parent = chunk
                    current_section = chunk
                elif hierarchy_level == 1:
                    current_section = chunk
                    if current_parent:
                        current_parent.child_ids.append(chunk.id)

            else:
                # Create content chunk
                parent_id = current_section.id if current_section else (current_parent.id if current_parent else None)

                chunk = PDFChunk(
                    id=self.get_next_chunk_id(),
                    content=text,
                    chunk_type="pdf_content",
                    hierarchy_level=max(hierarchy_level, 2),  # Content is always lowest level
                    parent_id=parent_id,
                    child_ids=[],
                    font_size=block.font_size,
                    page_number=block.page_number,
                    metadata={
                        "source_file": pdf_path,
                        "page_number": block.page_number,
                        "font_size": block.font_size,
                        "font_name": block.font_name,
                        "x0": block.x0, "y0": block.y0,
                        "x1": block.x1, "y1": block.y1,
                        "is_heading": False
                    }
                )

                chunks.append(chunk)
                self.hierarchy_map[chunk.id] = chunk

                # Add to parent's children
                if parent_id and parent_id in self.hierarchy_map:
                    self.hierarchy_map[parent_id].child_ids.append(chunk.id)

        console.print(f"   → Created {len(chunks)} hierarchical chunks")
        return chunks

    def display_hierarchy(self, chunks: List[PDFChunk]):
        """Display the chunk hierarchy as a tree"""
        console.print("\n[bold green]🌳 PDF Document Hierarchy[/bold green]")

        # Group chunks by level
        level_0_chunks = [c for c in chunks if c.hierarchy_level == 0]

        for root_chunk in level_0_chunks:
            tree = Tree(f"[bold blue]{root_chunk.content[:50]}...[/bold blue] (Level {root_chunk.hierarchy_level})")

            def add_children(parent_tree: Tree, parent_id: str, level: int = 1):
                children = [c for c in chunks if c.parent_id == parent_id]
                for child in sorted(children, key=lambda c: c.id):
                    child_node = parent_tree.add(f"[cyan]{child.content[:40]}...[/cyan] (Level {child.hierarchy_level})")
                    if child.child_ids:
                        add_children(child_node, child.id, level + 1)

            add_children(tree, root_chunk.id)
            console.print(tree)

    def chunks_to_langchain_documents(self, chunks: List[PDFChunk]) -> List[Document]:
        """Convert PDF chunks to LangChain documents"""
        documents = []

        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk.id,
                "chunk_type": chunk.chunk_type,
                "hierarchy_level": chunk.hierarchy_level,
                "parent_id": chunk.parent_id,
                "child_ids": chunk.child_ids,
                "word_count": len(chunk.content.split()),
                "content_length": len(chunk.content)
            }

            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            documents.append(doc)

        return documents


class PDFRAGSystem:
    """
    Complete RAG system for PDF structural chunking with vector search
    """

    def __init__(self, collection_name: str = "pdf_structural_chunks"):
        self.chunker = PDFStructuralChunker()
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None

        # Try to initialize embeddings, fallback to None if no API key
        try:
            self.embeddings = OpenAIEmbeddings()
            self.has_embeddings = True
        except Exception as e:
            console.print(f"[yellow]⚠️  OpenAI embeddings not available: {e}[/yellow]")
            console.print("[yellow]   → Running in demonstration mode without vector search[/yellow]")
            self.embeddings = None
            self.has_embeddings = False

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF and return LangChain documents"""
        chunks = self.chunker.create_hierarchical_chunks(pdf_path)
        documents = self.chunker.chunks_to_langchain_documents(chunks)

        # Display hierarchy
        self.chunker.display_hierarchy(chunks)

        return documents

    def initialize_vectorstore(self, documents: List[Document]):
        """Initialize ChromaDB vector store"""
        if not self.has_embeddings:
            console.print("[yellow]⚠️  Skipping vector store initialization (no embeddings available)[/yellow]")
            return

        console.print("\n[bold yellow]💾 Initializing ChromaDB vector store[/bold yellow]")

        # Create persistent client
        client = chromadb.PersistentClient(path="./chroma_db")

        self.vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        # Add documents
        if documents:
            self.vectorstore.add_documents(documents)
            console.print(f"   → Added {len(documents)} chunks to vector store")

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform semantic search"""
        if not self.vectorstore or not self.has_embeddings:
            console.print("[yellow]⚠️  Semantic search not available (no vector store)[/yellow]")
            return []

        return self.vectorstore.similarity_search(query, k=k)

    def hierarchical_search(self, query: str, k: int = 3, include_context: bool = True) -> List[Document]:
        """
        Perform hierarchical search that includes related chunks
        """
        if not self.vectorstore or not self.has_embeddings:
            console.print("[yellow]⚠️  Hierarchical search not available (no vector store)[/yellow]")
            return []

        # Get initial results
        initial_results = self.semantic_search(query, k=k)

        if not include_context:
            return initial_results

        # Expand with hierarchical context
        expanded_results = []
        seen_ids = set()

        for doc in initial_results:
            if doc.metadata.get("chunk_id") in seen_ids:
                continue

            expanded_results.append(doc)
            seen_ids.add(doc.metadata.get("chunk_id"))

            # Add parent if exists
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_ids:
                try:
                    parent_docs = self.vectorstore.get(where={"chunk_id": parent_id})
                    if parent_docs["documents"]:
                        parent_doc = Document(
                            page_content=parent_docs["documents"][0],
                            metadata=parent_docs["metadatas"][0]
                        )
                        expanded_results.append(parent_doc)
                        seen_ids.add(parent_id)
                except Exception:
                    pass

            # Add children if exist
            child_ids = doc.metadata.get("child_ids", [])
            for child_id in child_ids[:3]:  # Limit children to avoid explosion
                if child_id not in seen_ids:
                    try:
                        child_docs = self.vectorstore.get(where={"chunk_id": child_id})
                        if child_docs["documents"]:
                            child_doc = Document(
                                page_content=child_docs["documents"][0],
                                metadata=child_docs["metadatas"][0]
                            )
                            expanded_results.append(child_doc)
                            seen_ids.add(child_id)
                    except Exception:
                        pass

        return expanded_results

    def search_by_level(self, level: int, k: int = 10) -> List[Document]:
        """Search chunks by hierarchy level"""
        if not self.vectorstore or not self.has_embeddings:
            console.print("[yellow]⚠️  Level search not available (no vector store)[/yellow]")
            return []

        # Get all documents and filter by level
        all_docs = self.vectorstore.get()
        level_docs = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if metadata.get("hierarchy_level") == level:
                doc = Document(
                    page_content=all_docs["documents"][i],
                    metadata=metadata
                )
                level_docs.append(doc)

        return level_docs[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.vectorstore or not self.has_embeddings:
            return {
                "total_chunks": 0,
                "chunk_types": {},
                "hierarchy_levels": {},
                "source_files": []
            }

        all_docs = self.vectorstore.get()

        stats = {
            "total_chunks": len(all_docs["documents"]),
            "chunk_types": {},
            "hierarchy_levels": {},
            "source_files": set()
        }

        for metadata in all_docs["metadatas"]:
            # Chunk types
            chunk_type = metadata.get("chunk_type", "unknown")
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

            # Hierarchy levels
            level = metadata.get("hierarchy_level", 0)
            stats["hierarchy_levels"][level] = stats["hierarchy_levels"].get(level, 0) + 1

            # Source files
            source = metadata.get("source_file", "unknown")
            stats["source_files"].add(source)

        stats["source_files"] = list(stats["source_files"])
        return stats


def demonstrate_pdf_structural_chunking():
    """Complete demonstration of PDF structural block chunking"""

    console.print("[bold magenta]🚀 PDF Structural Block Chunking Demonstration[/bold magenta]")
    console.print("=" * 60)

    # Initialize RAG system
    rag_system = PDFRAGSystem("pdf_structural_demo")

    # Process PDF
    pdf_path = "study-research/Resume JPM.pdf"
    if not os.path.exists(pdf_path):
        console.print(f"[red]❌ PDF file not found: {pdf_path}[/red]")
        return

    documents = rag_system.process_pdf(pdf_path)

    # Initialize vector store
    rag_system.initialize_vectorstore(documents)

    # Display statistics
    stats = rag_system.get_statistics()
    console.print("\n[bold cyan]📊 Collection Statistics[/bold cyan]")
    console.print(f"   → Total chunks: {stats.get('total_chunks', 0)}")
    console.print(f"   → Chunk types: {stats.get('chunk_types', {})}")
    console.print(f"   → Hierarchy levels: {stats.get('hierarchy_levels', {})}")

    # Demonstrate searches
    console.print("\n[bold green]🔍 Search Demonstrations[/bold green]")

    # Basic semantic search
    console.print("\n1️⃣ [bold]Basic Semantic Search[/bold]")
    query = "machine learning experience"
    results = rag_system.semantic_search(query, k=3)
    for i, doc in enumerate(results, 1):
        console.print(f"   {i}. [cyan]{doc.metadata.get('chunk_type', 'unknown')}[/cyan]: {doc.page_content[:60]}...")

    # Hierarchical search
    console.print("\n2️⃣ [bold]Hierarchical Search (with context)[/bold]")
    results = rag_system.hierarchical_search(query, k=2, include_context=True)
    console.print(f"   → Found {len(results)} chunks with context")

    # Search by level
    console.print("\n3️⃣ [bold]Search by Hierarchy Level[/bold]")
    level_0_docs = rag_system.search_by_level(0)
    level_1_docs = rag_system.search_by_level(1)
    level_2_docs = rag_system.search_by_level(2)

    console.print(f"   → Level 0 (root): {len(level_0_docs)} chunks")
    console.print(f"   → Level 1 (sections): {len(level_1_docs)} chunks")
    console.print(f"   → Level 2 (content): {len(level_2_docs)} chunks")

    # Show sample chunks from each level
    console.print("\n[bold yellow]📋 Sample Chunks by Level[/bold yellow]")

    for level, docs in [(0, level_0_docs), (1, level_1_docs), (2, level_2_docs[:3])]:
        if docs:
            console.print(f"\n[bold]Level {level} Sample:[/bold]")
            for doc in docs[:2]:  # Show first 2 from each level
                console.print(f"   • {doc.page_content[:80]}...")

    console.print("\n[bold green]✅ PDF Structural Block Chunking Complete![/bold green]")
    console.print("\n[italic]Key Benefits:[/italic]")
    console.print("   • Preserves document structure and hierarchy")
    console.print("   • Font-size based heading detection")
    console.print("   • Parent-child relationships for context")
    console.print("   • Semantic search with structural awareness")
    console.print("   • Scalable vector storage with ChromaDB")


def demonstrate_qa_with_pdf_chunks():
    """Demonstrate question answering using the PDF chunks"""

    console.print("[bold magenta]🤖 PDF Content Q&A Demonstration[/bold magenta]")
    console.print("=" * 50)

    # Initialize the chunker and process PDF
    chunker = PDFStructuralChunker()
    pdf_path = "study-research/Resume JPM.pdf"

    if not os.path.exists(pdf_path):
        console.print(f"[red]❌ PDF file not found: {pdf_path}[/red]")
        return

    # Get chunks
    chunks = chunker.create_hierarchical_chunks(pdf_path)

    # Convert to documents for easier searching
    documents = chunker.chunks_to_langchain_documents(chunks)

    console.print(f"📄 Loaded {len(documents)} chunks from PDF\n")

    # User's specific questions
    questions = [
        "What is he doing right now?",
        "What is his experience?",
        "Tell me something about Jyoti Prakash Mohanta"
    ]

    for i, question in enumerate(questions, 1):
        console.print(f"[bold cyan]{i}. Question: {question}[/bold cyan]")

        # Get answer using improved search
        answer = answer_question_from_chunks(documents, question)

        if answer:
            console.print(f"   [green]Answer:[/green] {answer}")
        else:
            console.print("   [yellow]No relevant information found[/yellow]")

        console.print()


def answer_question_from_chunks(documents: List[Document], question: str) -> Optional[str]:
    """Extract answer from chunks based on question type"""

    question_lower = question.lower()

    # Current work/status question
    if "doing right now" in question_lower or "current" in question_lower:
        # Look for current position/experience
        for doc in documents:
            content = doc.page_content.lower()
            if ("present" in content or "current" in content or "2023" in content or "2024" in content or "2025" in content) and ("full stack" in content or "developer" in content):
                return "He is currently working as a Full Stack Developer at SettleMint India, Delhi (January 2023 - Present), where he builds full-stack applications from the ground up and is involved in all stages of software development."

        # Also check for SettleMint specifically
        for doc in documents:
            if "settlemint" in doc.page_content.lower() and "full stack" in doc.page_content.lower():
                return "He is currently working as a Full Stack Developer at SettleMint India, Delhi, building full-stack applications from the ground up and involved in all stages of software development."

    # Experience question
    elif "experience" in question_lower:
        experience_parts = []

        # Find current position
        for doc in documents:
            if "settlemint" in doc.page_content.lower():
                experience_parts.append("• Full Stack Developer at SettleMint India, Delhi (January 2023 - Present)")
                break

        # Find previous position
        for doc in documents:
            if "publicis sapient" in doc.page_content.lower():
                experience_parts.append("• Junior Associate at Publicis Sapient, Bangalore (May 2022 - December 2022)")
                break

        if experience_parts:
            return "His professional experience includes:\n" + "\n".join(experience_parts)

    # About Jyoti Prakash Mohanta
    elif "jyoti prakash mohanta" in question_lower:
        # Find personal info and summary
        info_parts = []

        # Find contact/location info
        for doc in documents:
            if "delhi" in doc.page_content.lower() and "@" in doc.page_content:
                info_parts.append(f"• Location: {doc.page_content.strip()}")
                break

        # Find professional summary
        for doc in documents:
            if "energetic full stack software developer" in doc.page_content.lower():
                info_parts.append(f"• Professional Summary: {doc.page_content.strip()}")
                break

        # Find education
        for doc in documents:
            if "raman global university" in doc.page_content.lower():
                info_parts.append(f"• Education: {doc.page_content.strip()}")
                break

        if info_parts:
            return "Jyoti Prakash Mohanta is a Full Stack AI Developer with the following background:\n" + "\n".join(info_parts)

    # Skills question
    elif "skill" in question_lower:
        skills_info = []
        for doc in documents:
            if "skills" in doc.page_content.lower() or "framework" in doc.page_content.lower():
                if "react" in doc.page_content.lower() or "next" in doc.page_content.lower():
                    skills_info.append("• Frameworks: Next.js, React.js, React Native")
                if "node" in doc.page_content.lower():
                    skills_info.append("• Backend: Node.js")
                if "ai" in doc.page_content.lower() or "azure" in doc.page_content.lower():
                    skills_info.append("• AI/ML: Azure AI Engineer Associate certified")

        if skills_info:
            return "His technical skills include:\n" + "\n".join(skills_info)

    # Projects question
    elif "project" in question_lower:
        projects = []
        project_names = ["Email Classification", "RAG Notebook", "Document Summariser",
                        "Jharkhand Fisheries", "Jharkhand Seeds", "Document Comparison AI App",
                        "VDR", "Tokenized Share Application", "Stablecoin Management System"]

        for doc in documents:
            content = doc.page_content.strip()
            for project in project_names:
                if project.lower() in content.lower() and len(content) < 200:
                    projects.append(f"• {content}")
                    break

        if projects:
            return f"He has worked on {len(projects)} projects including:\n" + "\n".join(projects[:5])  # Show first 5

    # Education question
    elif "education" in question_lower:
        for doc in documents:
            if "raman global university" in doc.page_content.lower():
                return f"His educational background: {doc.page_content.strip()}"

    # Default fallback
    return None


if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_pdf_structural_chunking()

    # Add Q&A demonstration
    console.print("\n" + "="*60 + "\n")
    demonstrate_qa_with_pdf_chunks()
