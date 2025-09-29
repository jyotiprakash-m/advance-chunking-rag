import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List
import re
from rich import print
from dotenv import load_dotenv

load_dotenv()

class StructuralBlockChunker:
    """
    A chunker that splits text into structural blocks based on document structure
    like headings, paragraphs, and other semantic units.
    """

    def __init__(self):
        self.blocks = []

    def split_into_structural_blocks(self, text: str) -> List[Document]:
        """
        Split text into structural blocks based on:
        - Headings (# ## ###)
        - Paragraphs (double newlines)
        - Lists and other structural elements
        """
        documents = []

        # Split by major structural elements
        # First, split by headings
        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)

        block_id = 0
        for section in sections:
            if not section.strip():
                continue

            # Check if this is a heading
            if re.match(r'^#{1,6}\s', section):
                # This is a heading section
                heading_match = re.match(r'^(#{1,6})\s*(.+?)$', section, re.MULTILINE)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    # Get content after heading
                    content_lines = section.split('\n', 1)
                    content = content_lines[1] if len(content_lines) > 1 else ""

                    # Split content into paragraphs
                    paragraphs = re.split(r'\n\s*\n', content.strip())

                    for para in paragraphs:
                        if para.strip():
                            doc = Document(
                                page_content=para.strip(),
                                metadata={
                                    "block_type": "paragraph",
                                    "heading_level": level,
                                    "heading_title": title,
                                    "block_id": block_id,
                                    "source": "structural_blocks"
                                }
                            )
                            documents.append(doc)
                            block_id += 1
            else:
                # This is content without a heading, treat as paragraphs
                paragraphs = re.split(r'\n\s*\n', section.strip())

                for para in paragraphs:
                    if para.strip():
                        doc = Document(
                            page_content=para.strip(),
                            metadata={
                                "block_type": "paragraph",
                                "heading_level": None,
                                "heading_title": None,
                                "block_id": block_id,
                                "source": "structural_blocks"
                            }
                        )
                        documents.append(doc)
                        block_id += 1

        return documents

def setup_chromadb_collection():
    """Setup ChromaDB client and collection"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="structural_blocks_demo",
        metadata={"description": "Demo collection for structural block chunking"}
    )
    return client, collection

def store_chunks_in_chromadb(chunks: List[Document], collection):
    """Store chunks in ChromaDB with embeddings"""
    # Initialize embeddings
    embedding_function = OpenAIEmbeddings()

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"block_{chunk.metadata['block_id']}"
        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)

    # Generate embeddings
    embeddings_list = embedding_function.embed_documents(documents)

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

def query_structural_blocks(query: str, collection, top_k: int = 3):
    """Query the structural blocks collection"""
    # Initialize embeddings for query
    embedding_function = OpenAIEmbeddings()

    # Generate query embedding
    query_embedding = embedding_function.embed_query(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    return results

def main():
    """Main demo function"""
    print("🚀 Structural Blocks Chunking with ChromaDB Demo")
    print("=" * 50)

    # Sample text with structural elements
    sample_text = """
# Introduction to Text Chunking

Text chunking is a critical technique in natural language processing and information retrieval. It involves breaking down large pieces of text into smaller, manageable units called chunks.

## Why Chunking Matters

Chunking helps improve efficiency and accuracy by dividing text into segments that are easier to analyze and process. This approach is especially useful in applications like information retrieval, summarization, and machine learning.

### Benefits of Structural Chunking

Structural chunking preserves the semantic relationships within the text by respecting document structure. Unlike fixed-size chunking, structural chunking:

- Maintains paragraph boundaries
- Preserves heading hierarchies
- Keeps related content together
- Improves retrieval relevance

## Types of Chunking Strategies

### Fixed-Size Chunking
This method splits text into chunks of predetermined sizes, regardless of content structure.

### Semantic Chunking
Semantic chunking groups text based on meaning and context, creating more intelligent divisions.

### Structural Chunking
Structural chunking, as demonstrated here, splits text based on document structure like headings and paragraphs.

## Conclusion

Choosing the right chunking strategy depends on your specific use case and requirements. Structural chunking is particularly effective for documents with clear hierarchical structure.
"""

    # Initialize chunker
    chunker = StructuralBlockChunker()

    # Split text into structural blocks
    print("📝 Splitting text into structural blocks...")
    chunks = chunker.split_into_structural_blocks(sample_text)

    print(f"✅ Created {len(chunks)} structural blocks")

    # Display sample chunks
    print("\n📋 Sample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nBlock {i+1}:")
        print(f"Type: {chunk.metadata['block_type']}")
        if chunk.metadata['heading_title']:
            print(f"Heading: {chunk.metadata['heading_title']} (Level {chunk.metadata['heading_level']})")
        print(f"Content: {chunk.page_content[:100]}...")

    # Setup ChromaDB
    print("\n🗄️ Setting up ChromaDB...")
    client, collection = setup_chromadb_collection()

    # Store chunks
    print("💾 Storing chunks in ChromaDB...")
    store_chunks_in_chromadb(chunks, collection)

    # Query examples
    queries = [
        "What is structural chunking?",
        "Benefits of chunking strategies",
        "How does semantic chunking work?"
    ]

    print("\n🔍 Querying ChromaDB...")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = query_structural_blocks(query, collection, top_k=2)

        if results and 'documents' in results and results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0] if results['documents'] else []
            metadatas = results.get('metadatas', [[]])
            metadatas = metadatas[0] if metadatas and len(metadatas) > 0 else []
            distances = results.get('distances', [[]])
            distances = distances[0] if distances and len(distances) > 0 else []

            for i, (doc, metadata, distance) in enumerate(zip(docs, metadatas, distances)):
                print(f"  Result {i+1} (distance: {distance:.3f}):")
                print(f"    {doc[:150]}...")
                if metadata and isinstance(metadata, dict) and metadata.get('heading_title'):
                    print(f"    From: {metadata['heading_title']}")
        else:
            print("  No results found")

    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()
