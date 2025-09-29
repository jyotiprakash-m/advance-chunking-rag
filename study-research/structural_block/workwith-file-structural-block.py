"""
File System Structural Block Chunking with Parent-Child Relationships
======================================================================

This script demonstrates how to work with files from the file system,
apply structural block chunking, and explore parent-child relationships
between chunks in a hierarchical document structure.

Now includes ChromaDB integration for vector storage and retrieval!
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from rich import print
from rich.tree import Tree
from rich.table import Table
from rich.console import Console
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

class HierarchicalChunker:
    """
    Advanced chunker that maintains parent-child relationships
    and hierarchical document structure.
    """

    def __init__(self):
        self.chunk_counter = 0
        self.hierarchy_map = {}  # Maps chunk IDs to their relationships

    def get_next_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        self.chunk_counter += 1
        return f"chunk_{self.chunk_counter:04d}"

    def create_chunk_document(self,
                            content: str,
                            chunk_type: str,
                            metadata: Dict[str, Any],
                            parent_id: Optional[str] = None) -> Document:
        """Create a document with hierarchical metadata"""

        chunk_id = self.get_next_chunk_id()

        # Enhanced metadata with hierarchy info
        enhanced_metadata = {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "content_length": len(content),
            "word_count": len(content.split()),
            "parent_id": parent_id,
            "child_ids": [],  # Will be populated later
            "hierarchy_level": 0,  # Will be calculated
            "source_file": metadata.get("source_file", "unknown"),
            **metadata
        }

        # Store hierarchy information
        self.hierarchy_map[chunk_id] = {
            "parent_id": parent_id,
            "children": [],
            "metadata": enhanced_metadata.copy()
        }

        # Update parent's children list if parent exists
        if parent_id and parent_id in self.hierarchy_map:
            self.hierarchy_map[parent_id]["children"].append(chunk_id)
            enhanced_metadata["child_ids"] = self.hierarchy_map[parent_id]["children"]

        return Document(
            page_content=content.strip(),
            metadata=enhanced_metadata
        )

    def calculate_hierarchy_levels(self):
        """Calculate hierarchy levels for all chunks"""
        def get_level(chunk_id: str, visited: Optional[set] = None) -> int:
            if visited is None:
                visited = set()

            if chunk_id in visited:
                return 0  # Prevent infinite recursion

            visited.add(chunk_id)

            parent_id = self.hierarchy_map[chunk_id]["parent_id"]
            if parent_id is None:
                return 0
            else:
                return get_level(parent_id, visited) + 1

        for chunk_id in self.hierarchy_map:
            level = get_level(chunk_id)
            self.hierarchy_map[chunk_id]["metadata"]["hierarchy_level"] = level

    def get_chunk_hierarchy(self) -> Dict[str, Any]:
        """Get the complete hierarchy map"""
        self.calculate_hierarchy_levels()
        return self.hierarchy_map

    def split_markdown_with_hierarchy(self, text: str, source_file: str) -> List[Document]:
        """
        Split markdown text while maintaining hierarchical relationships
        """
        documents = []

        # Split by major headings first (H1)
        h1_sections = re.split(r'(?=^# [^#])', text, flags=re.MULTILINE)

        for h1_section in h1_sections:
            if not h1_section.strip():
                continue

            if re.match(r'^# [^#]', h1_section):
                # This is an H1 section
                h1_match = re.match(r'^#\s*(.+?)$', h1_section, re.MULTILINE)
                if h1_match:
                    h1_title = h1_match.group(1).strip()

                    # Create H1 chunk
                    h1_content_lines = h1_section.split('\n', 1)
                    h1_content = h1_content_lines[1] if len(h1_content_lines) > 1 else ""

                    h1_doc = self.create_chunk_document(
                        h1_content,
                        "h1_section",
                        {
                            "heading_level": 1,
                            "heading_title": h1_title,
                            "source_file": source_file
                        }
                    )
                    documents.append(h1_doc)
                    h1_id = h1_doc.metadata["chunk_id"]

                    # Now split H1 content by H2 headings
                    h2_sections = re.split(r'(?=^## [^#])', h1_content, flags=re.MULTILINE)

                    for h2_section in h2_sections:
                        if not h2_section.strip():
                            continue

                        if re.match(r'^## [^#]', h2_section):
                            # This is an H2 section
                            h2_match = re.match(r'^##\s*(.+?)$', h2_section, re.MULTILINE)
                            if h2_match:
                                h2_title = h2_match.group(1).strip()

                                # Create H2 chunk
                                h2_content_lines = h2_section.split('\n', 1)
                                h2_content = h2_content_lines[1] if len(h2_content_lines) > 1 else ""

                                h2_doc = self.create_chunk_document(
                                    h2_content,
                                    "h2_section",
                                    {
                                        "heading_level": 2,
                                        "heading_title": h2_title,
                                        "source_file": source_file
                                    },
                                    parent_id=h1_id
                                )
                                documents.append(h2_doc)
                                h2_id = h2_doc.metadata["chunk_id"]

                                # Split H2 content by H3 headings
                                h3_sections = re.split(r'(?=^### [^#])', h2_content, flags=re.MULTILINE)

                                for h3_section in h3_sections:
                                    if not h3_section.strip():
                                        continue

                                    if re.match(r'^### [^#]', h3_section):
                                        # This is an H3 section
                                        h3_match = re.match(r'^###\s*(.+?)$', h3_section, re.MULTILINE)
                                        if h3_match:
                                            h3_title = h3_match.group(1).strip()

                                            # Create H3 chunk
                                            h3_content_lines = h3_section.split('\n', 1)
                                            h3_content = h3_content_lines[1] if len(h3_content_lines) > 1 else ""

                                            h3_doc = self.create_chunk_document(
                                                h3_content,
                                                "h3_section",
                                                {
                                                    "heading_level": 3,
                                                    "heading_title": h3_title,
                                                    "source_file": source_file
                                                },
                                                parent_id=h2_id
                                            )
                                            documents.append(h3_doc)
                                    else:
                                        # Content under H2 without H3
                                        if h3_section.strip():
                                            content_doc = self.create_chunk_document(
                                                h3_section,
                                                "h2_content",
                                                {
                                                    "heading_level": 2,
                                                    "heading_title": h2_title,
                                                    "source_file": source_file
                                                },
                                                parent_id=h2_id
                                            )
                                            documents.append(content_doc)
                        else:
                            # Content under H1 without H2
                            if h2_section.strip():
                                content_doc = self.create_chunk_document(
                                    h2_section,
                                    "h1_content",
                                    {
                                        "heading_level": 1,
                                        "heading_title": h1_title,
                                        "source_file": source_file
                                    },
                                    parent_id=h1_id
                                )
                                documents.append(content_doc)
            else:
                # Content without H1
                if h1_section.strip():
                    content_doc = self.create_chunk_document(
                        h1_section,
                        "root_content",
                        {
                            "heading_level": 0,
                            "source_file": source_file
                        }
                    )
                    documents.append(content_doc)

        return documents

class FileSystemChunker:
    """
    Works with file system to read files and apply structural chunking
    """

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.hierarchical_chunker = HierarchicalChunker()
        self.processed_files = []

    def read_file(self, file_path: str) -> Optional[str]:
        """Read a file and return its content"""
        try:
            full_path = self.base_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            return None

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        full_path = self.base_path / file_path
        stat = full_path.stat()

        return {
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "modified_time": stat.st_mtime,
            "extension": full_path.suffix,
            "name": full_path.name
        }

    def process_file(self, file_path: str) -> List[Document]:
        """Process a single file with structural chunking"""
        print(f"\n📄 Processing file: {file_path}")

        # Read file content
        content = self.read_file(file_path)
        if content is None:
            return []

        # Get file info
        file_info = self.get_file_info(file_path)

        # Determine file type and apply appropriate chunking
        if file_path.endswith('.md') or file_path.endswith('.markdown'):
            print("   → Detected Markdown file, applying hierarchical chunking")
            chunks = self.hierarchical_chunker.split_markdown_with_hierarchy(
                content, file_path
            )
        elif file_path.endswith('.html') or file_path.endswith('.htm'):
            print("   → Detected HTML file, applying header-based chunking")
            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
            chunks = splitter.split_text(content)
            # Convert to our format
            chunks = [
                self.hierarchical_chunker.create_chunk_document(
                    chunk.page_content,
                    "html_section",
                    {
                        "source_file": file_path,
                        **file_info
                    }
                ) for chunk in chunks
            ]
        else:
            print("   → Generic file, applying recursive character chunking")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.create_documents([content])
            # Convert to our format
            chunks = [
                self.hierarchical_chunker.create_chunk_document(
                    chunk.page_content,
                    "generic_chunk",
                    {
                        "source_file": file_path,
                        **file_info
                    }
                ) for chunk in chunks
            ]

        self.processed_files.append({
            "file_path": file_path,
            "chunks_count": len(chunks),
            "file_info": file_info
        })

        print(f"   ✅ Created {len(chunks)} chunks")
        return chunks

    def process_directory(self, directory_path: str = ".", extensions: Optional[List[str]] = None) -> List[Document]:
        """Process all files in a directory"""
        if extensions is None:
            extensions = ['.md', '.markdown', '.txt', '.html', '.htm']

        all_chunks = []
        directory = self.base_path / directory_path

        print(f"\n📁 Processing directory: {directory_path}")
        print(f"   → Looking for files with extensions: {extensions}")

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                relative_path = file_path.relative_to(self.base_path)
                chunks = self.process_file(str(relative_path))
                all_chunks.extend(chunks)

        print(f"\n📊 Directory processing complete:")
        print(f"   → Total files processed: {len(self.processed_files)}")
        print(f"   → Total chunks created: {len(all_chunks)}")

        return all_chunks

    def get_hierarchy_tree(self) -> Tree:
        """Generate a visual tree of chunk relationships"""
        hierarchy = self.hierarchical_chunker.get_chunk_hierarchy()

        # Find root chunks (no parent)
        root_chunks = [cid for cid, data in hierarchy.items() if data["parent_id"] is None]

        tree = Tree("📚 Document Hierarchy", style="bold blue")

        def add_chunk_to_tree(chunk_id: str, tree_node: Tree):
            chunk_data = hierarchy[chunk_id]
            metadata = chunk_data["metadata"]

            # Create node label
            level_indicator = "🏠" if metadata["hierarchy_level"] == 0 else "📄"
            type_indicator = metadata.get("chunk_type", "unknown")[:3].upper()
            title = metadata.get("heading_title", f"Chunk {chunk_id}")

            node_label = f"{level_indicator} {type_indicator}: {title[:30]}..."
            if len(title) > 30:
                node_label = f"{level_indicator} {type_indicator}: {title[:27]}..."

            # Add word count and level info
            node_label += f" ({metadata['word_count']} words, Level {metadata['hierarchy_level']})"

            branch = tree_node.add(node_label)

            # Add children
            for child_id in chunk_data["children"]:
                add_chunk_to_tree(child_id, branch)

        for root_id in root_chunks:
            add_chunk_to_tree(root_id, tree)

        return tree

    def get_relationships_table(self) -> Table:
        """Generate a table showing chunk relationships"""
        hierarchy = self.hierarchical_chunker.get_chunk_hierarchy()

        table = Table(title="🔗 Chunk Relationships")
        table.add_column("Chunk ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Level", style="yellow", justify="center")
        table.add_column("Parent", style="red", no_wrap=True)
        table.add_column("Children", style="green", justify="center")
        table.add_column("Words", style="blue", justify="right")

        for chunk_id, data in hierarchy.items():
            metadata = data["metadata"]
            children_count = len(data["children"])
            parent_id = data["parent_id"] or "ROOT"

            table.add_row(
                chunk_id,
                metadata.get("chunk_type", "unknown"),
                str(metadata.get("hierarchy_level", 0)),
                parent_id,
                str(children_count),
                str(metadata.get("word_count", 0))
            )

        return table

    def analyze_hierarchy_stats(self) -> Dict[str, Any]:
        """Analyze hierarchy statistics"""
        hierarchy = self.hierarchical_chunker.get_chunk_hierarchy()

        stats = {
            "total_chunks": len(hierarchy),
            "max_depth": 0,
            "avg_children_per_parent": 0,
            "chunk_types": {},
            "hierarchy_levels": {}
        }

        total_children = 0
        parent_count = 0

        for chunk_id, data in hierarchy.items():
            metadata = data["metadata"]
            children = data["children"]

            # Track depth
            level = metadata.get("hierarchy_level", 0)
            stats["max_depth"] = max(stats["max_depth"], level)

            # Track levels
            level_key = str(level)
            stats["hierarchy_levels"][level_key] = stats["hierarchy_levels"].get(level_key, 0) + 1

            # Track types
            chunk_type = metadata.get("chunk_type", "unknown")
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

            # Track children
            if children:
                total_children += len(children)
                parent_count += 1

        if parent_count > 0:
            stats["avg_children_per_parent"] = total_children / parent_count

        return stats

class ChromaDBManager:
    """
    Manages ChromaDB operations for hierarchical chunks
    """

    def __init__(self, collection_name: str = "hierarchical_chunks"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.console = Console()

    def initialize_db(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB vector store"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        print(f"✅ Initialized ChromaDB collection: {self.collection_name}")

    def add_chunks_to_db(self, chunks: List[Document], hierarchy_map: Dict[str, Any]):
        """Add chunks to ChromaDB with enhanced metadata"""
        if self.vectorstore is None:
            self.initialize_db()

        # Prepare documents with enhanced metadata
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            # Get hierarchy information
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                continue  # Skip chunks without IDs

            hierarchy_info = hierarchy_map.get(chunk_id, {})

            # Enhanced metadata for retrieval (ChromaDB compatible)
            enhanced_metadata = {
                **chunk.metadata,
                "parent_id": hierarchy_info.get("parent_id", ""),
                "child_ids": json.dumps(hierarchy_info.get("children", [])),  # Convert list to JSON string
                "hierarchy_level": hierarchy_info.get("metadata", {}).get("hierarchy_level", 0),
                "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                "word_count": chunk.metadata.get("word_count", 0),
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "heading_title": chunk.metadata.get("heading_title", ""),
                "content_length": len(chunk.page_content)
            }

            documents.append(chunk.page_content)
            metadatas.append(enhanced_metadata)
            ids.append(chunk_id)

        # Add to vector store
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.vectorstore.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Added {len(chunks)} chunks to ChromaDB")
        self.vectorstore.persist()

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def hierarchical_search(self, query: str, k: int = 3, include_parent: bool = True, include_children: bool = True) -> Dict[str, Any]:
        """
        Perform hierarchical search that includes related chunks
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        # Get initial similar chunks
        initial_results = self.similarity_search(query, k=k)

        result_chunks = []
        processed_ids = set()

        for chunk in initial_results:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id in processed_ids:
                continue

            result_chunks.append(chunk)
            processed_ids.add(chunk_id)

            # Include parent if requested
            if include_parent:
                parent_id = chunk.metadata.get("parent_id")
                if parent_id and parent_id not in processed_ids:
                    try:
                        parent_docs = self.vectorstore.get(where={"chunk_id": parent_id})
                        if parent_docs["documents"]:
                            parent_doc = Document(
                                page_content=parent_docs["documents"][0],
                                metadata=parent_docs["metadatas"][0]
                            )
                            result_chunks.append(parent_doc)
                            processed_ids.add(parent_id)
                    except Exception as e:
                        print(f"Warning: Could not retrieve parent {parent_id}: {e}")

            # Include children if requested
            if include_children:
                child_ids_str = chunk.metadata.get("child_ids", "[]")
                try:
                    child_ids = json.loads(child_ids_str) if child_ids_str else []
                except json.JSONDecodeError:
                    child_ids = []

                for child_id in child_ids:
                    if child_id not in processed_ids:
                        try:
                            child_docs = self.vectorstore.get(where={"chunk_id": child_id})
                            if child_docs["documents"]:
                                child_doc = Document(
                                    page_content=child_docs["documents"][0],
                                    metadata=child_docs["metadatas"][0]
                                )
                                result_chunks.append(child_doc)
                                processed_ids.add(child_id)
                        except Exception as e:
                            print(f"Warning: Could not retrieve child {child_id}: {e}")

        return {
            "query": query,
            "initial_results": len(initial_results),
            "total_results": len(result_chunks),
            "chunks": result_chunks
        }

    def search_by_hierarchy_level(self, level: int, k: int = 10) -> List[Document]:
        """Search chunks by hierarchy level"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.similarity_search(
            "",  # Empty query to get all
            k=k,
            filter={"hierarchy_level": str(level)}
        )

    def search_by_chunk_type(self, chunk_type: str, k: int = 10) -> List[Document]:
        """Search chunks by type"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.similarity_search(
            "",  # Empty query
            k=k,
            filter={"chunk_type": chunk_type}
        )

    def get_chunk_with_context(self, chunk_id: str) -> Dict[str, Any]:
        """Get a chunk with its full hierarchical context"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        # Get the main chunk
        try:
            chunk_docs = self.vectorstore.get(where={"chunk_id": chunk_id})
            if not chunk_docs["documents"]:
                return {"error": f"Chunk {chunk_id} not found"}

            main_chunk = Document(
                page_content=chunk_docs["documents"][0],
                metadata=chunk_docs["metadatas"][0]
            )
        except Exception as e:
            return {"error": f"Could not retrieve chunk {chunk_id}: {e}"}

        context = {
            "main_chunk": main_chunk,
            "parent": None,
            "children": [],
            "siblings": []
        }

        # Get parent
        parent_id = main_chunk.metadata.get("parent_id")
        if parent_id:
            try:
                parent_docs = self.vectorstore.get(where={"chunk_id": parent_id})
                if parent_docs["documents"]:
                    context["parent"] = Document(
                        page_content=parent_docs["documents"][0],
                        metadata=parent_docs["metadatas"][0]
                    )
            except Exception as e:
                print(f"Warning: Could not retrieve parent: {e}")

        # Get children
        child_ids_str = main_chunk.metadata.get("child_ids", "[]")
        try:
            child_ids = json.loads(child_ids_str) if child_ids_str else []
        except json.JSONDecodeError:
            child_ids = []

        for child_id in child_ids:
            try:
                child_docs = self.vectorstore.get(where={"chunk_id": child_id})
                if child_docs["documents"]:
                    context["children"].append(Document(
                        page_content=child_docs["documents"][0],
                        metadata=child_docs["metadatas"][0]
                    ))
            except Exception as e:
                print(f"Warning: Could not retrieve child {child_id}: {e}")

        # Get siblings (other children of the same parent)
        if parent_id:
            parent_metadata = context["parent"].metadata if context["parent"] else None
            if parent_metadata:
                siblings_str = parent_metadata.get("child_ids", "[]")
                try:
                    all_sibling_ids = json.loads(siblings_str) if siblings_str else []
                except json.JSONDecodeError:
                    all_sibling_ids = []

                for sibling_id in all_sibling_ids:
                    if sibling_id != chunk_id:
                        try:
                            sibling_docs = self.vectorstore.get(where={"chunk_id": sibling_id})
                            if sibling_docs["documents"]:
                                context["siblings"].append(Document(
                                    page_content=sibling_docs["documents"][0],
                                    metadata=sibling_docs["metadatas"][0]
                                ))
                        except Exception as e:
                            print(f"Warning: Could not retrieve sibling {sibling_id}: {e}")

        return context

    def create_qa_chain(self) -> RetrievalQA:
        """Create a QA chain for question answering"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return qa_chain

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.vectorstore is None:
            return {"error": "Vector store not initialized"}

        try:
            # Get all documents
            all_docs = self.vectorstore.get()
            total_chunks = len(all_docs["ids"])

            # Analyze metadata
            chunk_types = {}
            hierarchy_levels = {}
            source_files = {}

            for metadata in all_docs["metadatas"]:
                # Count chunk types
                chunk_type = metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                # Count hierarchy levels
                level = metadata.get("hierarchy_level", 0)
                hierarchy_levels[level] = hierarchy_levels.get(level, 0) + 1

                # Count source files
                source_file = metadata.get("source_file", "unknown")
                source_files[source_file] = source_files.get(source_file, 0) + 1

            return {
                "total_chunks": total_chunks,
                "chunk_types": chunk_types,
                "hierarchy_levels": hierarchy_levels,
                "source_files": source_files,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": f"Could not get stats: {e}"}

    def clear_collection(self):
        """Clear all documents from the collection"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print("✅ Cleared ChromaDB collection")

def demonstrate_file_system_chunking():
    """Demonstrate file system chunking with parent-child relationships"""

    print("🚀 File System Structural Block Chunking Demo")
    print("=" * 60)

    # Initialize chunker
    chunker = FileSystemChunker(".")

    # Process files in the study-research directory
    print("\n📂 Processing files in study-research directory...")

    # First, let's check what files are available
    study_dir = Path("./study-research")
    if study_dir.exists():
        available_files = []
        for ext in ['.md', '.markdown', '.txt', '.html', '.htm']:
            for file_path in study_dir.rglob(f'*{ext}'):
                available_files.append(file_path.relative_to("."))

        if available_files:
            print(f"📋 Found {len(available_files)} processable files:")
            for file_path in available_files[:5]:  # Show first 5
                print(f"   • {file_path}")
            if len(available_files) > 5:
                print(f"   ... and {len(available_files) - 5} more")

            # Process a sample file if it exists
            sample_file = "study-research/content.txt"
            if Path(sample_file).exists():
                print(f"\n🎯 Processing sample file: {sample_file}")
                chunks = chunker.process_file(sample_file)

                print(f"\n📊 Sample file results:")
                print(f"   → Created {len(chunks)} chunks")

                # Show first few chunks
                for i, chunk in enumerate(chunks[:3]):
                    print(f"\n   Chunk {i+1}:")
                    print(f"     ID: {chunk.metadata.get('chunk_id')}")
                    print(f"     Type: {chunk.metadata.get('chunk_type')}")
                    print(f"     Words: {chunk.metadata.get('word_count')}")
                    print(f"     Content: {chunk.page_content[:100]}...")
        else:
            print("❌ No processable files found in study-research directory")
    else:
        print("❌ study-research directory not found")

    # Create a sample markdown file for demonstration
    print("\n📝 Creating sample hierarchical markdown file...")

    sample_content = """
# Machine Learning Guide

## Introduction
Machine learning is transforming how we solve complex problems.

### What is ML?
ML enables computers to learn patterns from data without explicit programming.

### Why It Matters
ML applications span from recommendation systems to autonomous vehicles.

## Core Concepts

### Supervised Learning
Supervised learning uses labeled training data.

#### Classification
Classification predicts categorical outcomes.

#### Regression
Regression predicts continuous values.

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data.

#### Clustering
Clustering groups similar data points.

#### Dimensionality Reduction
Reduces feature space while preserving information.

## Advanced Topics

### Deep Learning
Deep learning uses neural networks with multiple layers.

### Natural Language Processing
NLP enables computers to understand and generate human language.

## Conclusion
Machine learning continues to evolve and impact various industries.
"""

    sample_file_path = "study-research/sample_hierarchy.md"
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    print(f"✅ Created sample file: {sample_file_path}")

    # Process the sample file
    print(f"\n🔄 Processing sample hierarchical file...")
    chunks = chunker.process_file(sample_file_path)

    # Get hierarchy information
    hierarchy = chunker.hierarchical_chunker.get_chunk_hierarchy()

    print(f"\n🏗️ Hierarchy Analysis:")
    print(f"   → Total chunks: {len(hierarchy)}")

    # Show hierarchy tree
    print("\n🌳 Document Hierarchy Tree:")
    tree = chunker.get_hierarchy_tree()
    print(tree)

    # Show relationships table
    print("\n📋 Chunk Relationships Table:")
    table = chunker.get_relationships_table()
    print(table)

    # Show statistics
    stats = chunker.analyze_hierarchy_stats()
    print(f"\n📈 Hierarchy Statistics:")
    print(f"   → Total chunks: {stats['total_chunks']}")
    print(f"   → Maximum depth: {stats['max_depth']}")
    print(".2f")
    print(f"   → Chunk types: {stats['chunk_types']}")
    print(f"   → Chunks by level: {stats['hierarchy_levels']}")

    # Demonstrate parent-child navigation
    print(f"\n🔍 Parent-Child Navigation Examples:")

    # Find a chunk with children
    chunk_with_children = None
    for chunk_id, data in hierarchy.items():
        if data["children"]:
            chunk_with_children = chunk_id
            break

    if chunk_with_children:
        chunk_data = hierarchy[chunk_with_children]
        metadata = chunk_data["metadata"]

        print(f"\n   Parent Chunk: {chunk_with_children}")
        print(f"   → Type: {metadata.get('chunk_type')}")
        print(f"   → Title: {metadata.get('heading_title', 'N/A')}")
        print(f"   → Level: {metadata.get('hierarchy_level')}")
        print(f"   → Children: {len(chunk_data['children'])}")

        for i, child_id in enumerate(chunk_data["children"][:3]):
            child_data = hierarchy[child_id]
            child_metadata = child_data["metadata"]
            print(f"     Child {i+1}: {child_id} ({child_metadata.get('chunk_type')}) - '{child_metadata.get('heading_title', 'N/A')}'")

    print(f"\n🎉 File system chunking demonstration complete!")
    print(f"💡 Key insights about parent-child relationships:")
    print(f"   • Hierarchical chunking preserves document structure")
    print(f"   • Parent chunks contain metadata about their children")
    print(f"   • Child chunks reference their parent for context")
    print(f"   • This enables intelligent retrieval and navigation")

def demonstrate_chromadb_integration():
    """Demonstrate ChromaDB integration with hierarchical chunks"""

    print("🗄️ ChromaDB Integration Demo")
    print("=" * 50)

    # Initialize components
    chunker = FileSystemChunker(".")
    chroma_manager = ChromaDBManager("hierarchical_demo")

    # Process files and get chunks
    print("\n📄 Processing files...")
    chunks = chunker.process_file("study-research/sample_hierarchy.md")

    if not chunks:
        print("❌ No chunks created, cannot proceed with demo")
        return

    # Get hierarchy map
    hierarchy_map = chunker.hierarchical_chunker.get_chunk_hierarchy()

    # Add chunks to ChromaDB
    print("\n💾 Adding chunks to ChromaDB...")
    chroma_manager.add_chunks_to_db(chunks, hierarchy_map)

    # Show collection stats
    print("\n📊 Collection Statistics:")
    stats = chroma_manager.get_collection_stats()
    if "error" not in stats:
        print(f"   → Total chunks: {stats['total_chunks']}")
        print(f"   → Chunk types: {stats['chunk_types']}")
        print(f"   → Hierarchy levels: {stats['hierarchy_levels']}")
        print(f"   → Source files: {list(stats['source_files'].keys())}")

    # Demonstrate different retrieval methods
    print("\n🔍 Retrieval Demonstrations:")
    print("-" * 30)

    # 1. Basic similarity search
    print("\n1️⃣ Basic Similarity Search")
    query = "machine learning applications"
    results = chroma_manager.similarity_search(query, k=3)

    print(f"Query: '{query}'")
    print(f"Results: {len(results)} chunks")
    for i, result in enumerate(results, 1):
        metadata = result.metadata
        print(f"   {i}. {metadata.get('chunk_type', 'unknown')}: {result.page_content[:80]}...")
        print(f"      Level: {metadata.get('hierarchy_level')}, Words: {metadata.get('word_count')}")

    # 2. Hierarchical search
    print("\n2️⃣ Hierarchical Search (with parent/child context)")
    hierarchical_results = chroma_manager.hierarchical_search(query, k=2, include_parent=True, include_children=True)

    print(f"Query: '{query}'")
    print(f"Initial results: {hierarchical_results['initial_results']}")
    print(f"Total with context: {hierarchical_results['total_results']}")

    for i, chunk in enumerate(hierarchical_results['chunks'][:5], 1):
        metadata = chunk.metadata
        chunk_type = metadata.get('chunk_type', 'unknown')
        level = metadata.get('hierarchy_level', 0)
        title = metadata.get('heading_title', '')[:30]
        print(f"   {i}. [{level}] {chunk_type}: {title}... ({metadata.get('word_count')} words)")

    # 3. Search by hierarchy level
    print("\n3️⃣ Search by Hierarchy Level")
    level_2_chunks = chroma_manager.search_by_hierarchy_level(2, k=4)
    print(f"Level 2 chunks: {len(level_2_chunks)}")
    for chunk in level_2_chunks:
        metadata = chunk.metadata
        title = metadata.get('heading_title', 'No title')
        print(f"   • {title} ({metadata.get('word_count')} words)")

    # 4. Search by chunk type
    print("\n4️⃣ Search by Chunk Type")
    h3_chunks = chroma_manager.search_by_chunk_type("h3_section", k=3)
    print(f"H3 sections: {len(h3_chunks)}")
    for chunk in h3_chunks:
        metadata = chunk.metadata
        title = metadata.get('heading_title', 'No title')
        print(f"   • {title}")

    # 5. Get chunk with full context
    print("\n5️⃣ Get Chunk with Full Hierarchical Context")
    if chunks:
        sample_chunk_id = chunks[0].metadata.get("chunk_id")
        if sample_chunk_id:
            context = chroma_manager.get_chunk_with_context(sample_chunk_id)

            if "error" not in context:
                main_chunk = context["main_chunk"]
                print(f"Main chunk: {main_chunk.metadata.get('heading_title', 'No title')}")
                print(f"   Type: {main_chunk.metadata.get('chunk_type')}")
                print(f"   Level: {main_chunk.metadata.get('hierarchy_level')}")

                if context["parent"]:
                    parent = context["parent"]
                    print(f"   Parent: {parent.metadata.get('heading_title', 'No title')}")

                if context["children"]:
                    print(f"   Children: {len(context['children'])}")
                    for child in context["children"][:2]:
                        print(f"     • {child.metadata.get('heading_title', 'No title')}")

                if context["siblings"]:
                    print(f"   Siblings: {len(context['siblings'])}")

    # 6. QA Chain demonstration
    print("\n6️⃣ Question Answering with Retrieval")
    try:
        qa_chain = chroma_manager.create_qa_chain()

        questions = [
            "What is machine learning?",
            "What are the core concepts discussed?",
            "What advanced topics are covered?"
        ]

        for question in questions:
            print(f"\nQ: {question}")
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = len(result["source_documents"])

            print(f"A: {answer[:150]}..." if len(answer) > 150 else f"A: {answer}")
            print(f"   Sources: {sources} chunks")

    except Exception as e:
        print(f"❌ QA Chain error: {e}")

    print("\n🎉 ChromaDB integration demonstration complete!")
    print("\n💡 Key benefits of hierarchical chunking with ChromaDB:")
    print("   • Preserves document structure in vector search")
    print("   • Enables context-aware retrieval (parent/child relationships)")
    print("   • Supports multi-level filtering and navigation")
    print("   • Improves semantic search with structural awareness")

def main():
    """Main function"""
    # Run the original file system chunking demo
    demonstrate_file_system_chunking()

    # Add ChromaDB integration demo
    print("\n" + "="*80)
    demonstrate_chromadb_integration()

if __name__ == "__main__":
    main()
