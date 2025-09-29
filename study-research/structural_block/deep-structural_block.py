"""
Deep Dive into Structural Block Chunking
=========================================

This file explores advanced structural block chunking techniques,
demonstrating various document structures and chunking strategies.
"""

import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Any
import re
from rich import print
from dotenv import load_dotenv
import json

load_dotenv()

class AdvancedStructuralChunker:
    """
    Advanced chunker that handles complex document structures including:
    - Markdown headings and subheadings
    - Code blocks
    - Lists (ordered/unordered)
    - Tables
    - Blockquotes
    - Custom structural patterns
    """

    def __init__(self):
        self.chunk_id_counter = 0

    def get_next_chunk_id(self) -> int:
        """Get next unique chunk ID"""
        self.chunk_id_counter += 1
        return self.chunk_id_counter - 1

    def create_document(self, content: str, block_type: str, metadata: Dict[str, Any]) -> Document:
        """Create a Document with enhanced metadata"""
        base_metadata = {
            "block_type": block_type,
            "source": "deep_structural_blocks",
            "chunk_id": self.get_next_chunk_id(),
            "content_length": len(content),
            "word_count": len(content.split())
        }
        base_metadata.update(metadata)

        return Document(
            page_content=content.strip(),
            metadata=base_metadata
        )

    def split_markdown_headings(self, text: str) -> List[Document]:
        """
        Example 1: Basic heading-based chunking
        Splits text by Markdown headings, preserving hierarchy
        """
        print("📖 Example 1: Basic Heading-Based Chunking")

        documents = []
        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            if re.match(r'^#{1,6}\s', section):
                heading_match = re.match(r'^(#{1,6})\s*(.+?)$', section, re.MULTILINE)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    content_lines = section.split('\n', 1)
                    content = content_lines[1] if len(content_lines) > 1 else ""

                    if content.strip():
                        doc = self.create_document(
                            content,
                            "heading_section",
                            {
                                "heading_level": level,
                                "heading_title": title,
                                "hierarchy_path": f"{'#' * level} {title}"
                            }
                        )
                        documents.append(doc)

        print(f"   → Created {len(documents)} heading-based chunks")
        return documents

    def split_code_blocks(self, text: str) -> List[Document]:
        """
        Example 2: Code block extraction
        Separates code blocks from regular text for better processing
        """
        print("💻 Example 2: Code Block Extraction")

        documents = []

        # Split by code blocks (```language ... ```)
        parts = re.split(r'(```[\w]*\n.*?\n```)', text, flags=re.DOTALL)

        for part in parts:
            if not part.strip():
                continue

            if re.match(r'```[\w]*\n', part):
                # This is a code block
                code_match = re.match(r'```([\w]*)\n(.*?)\n```', part, re.DOTALL)
                if code_match:
                    language = code_match.group(1) or "text"
                    code_content = code_match.group(2)

                    doc = self.create_document(
                        code_content,
                        "code_block",
                        {
                            "language": language,
                            "is_code": True,
                            "lines_of_code": len(code_content.split('\n'))
                        }
                    )
                    documents.append(doc)
            else:
                # Regular text, split into paragraphs
                paragraphs = re.split(r'\n\s*\n', part.strip())
                for para in paragraphs:
                    if para.strip():
                        doc = self.create_document(
                            para,
                            "text_paragraph",
                            {"is_code": False}
                        )
                        documents.append(doc)

        print(f"   → Created {len(documents)} chunks (code + text)")
        return documents

    def split_lists_and_tables(self, text: str) -> List[Document]:
        """
        Example 3: List and table processing
        Handles ordered/unordered lists and Markdown tables
        """
        print("📋 Example 3: List and Table Processing")

        documents = []

        # Split by different structural elements
        parts = re.split(r'((?:^\d+\.\s.*$|^-\s.*$|^\|.*\|\s*$)+)', text, flags=re.MULTILINE)

        for part in parts:
            if not part.strip():
                continue

            if re.match(r'^\d+\.\s', part, re.MULTILINE):
                # Ordered list
                doc = self.create_document(
                    part,
                    "ordered_list",
                    {
                        "list_type": "ordered",
                        "items_count": len(re.findall(r'^\d+\.\s', part, re.MULTILINE))
                    }
                )
                documents.append(doc)

            elif re.match(r'^-\s', part, re.MULTILINE):
                # Unordered list
                doc = self.create_document(
                    part,
                    "unordered_list",
                    {
                        "list_type": "unordered",
                        "items_count": len(re.findall(r'^-\s', part, re.MULTILINE))
                    }
                )
                documents.append(doc)

            elif re.match(r'^\|.*\|\s*$', part, re.MULTILINE):
                # Table
                lines = part.strip().split('\n')
                doc = self.create_document(
                    part,
                    "table",
                    {
                        "table_rows": len(lines),
                        "is_table": True
                    }
                )
                documents.append(doc)
            else:
                # Regular text
                paragraphs = re.split(r'\n\s*\n', part.strip())
                for para in paragraphs:
                    if para.strip():
                        doc = self.create_document(
                            para,
                            "text_paragraph",
                            {"is_structured": False}
                        )
                        documents.append(doc)

        print(f"   → Created {len(documents)} chunks (lists, tables, text)")
        return documents

    def hierarchical_chunking(self, text: str) -> List[Document]:
        """
        Example 4: Hierarchical chunking with parent-child relationships
        Creates chunks that maintain document hierarchy
        """
        print("🏗️ Example 4: Hierarchical Chunking")

        documents = []
        hierarchy_stack = []

        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            if re.match(r'^#{1,6}\s', section):
                heading_match = re.match(r'^(#{1,6})\s*(.+?)$', section, re.MULTILINE)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    # Update hierarchy stack
                    while hierarchy_stack and hierarchy_stack[-1]['level'] >= level:
                        hierarchy_stack.pop()

                    hierarchy_stack.append({'level': level, 'title': title})

                    content_lines = section.split('\n', 1)
                    content = content_lines[1] if len(content_lines) > 1 else ""

                    if content.strip():
                        # Create hierarchy path
                        hierarchy_path = " > ".join([item['title'] for item in hierarchy_stack])

                        doc = self.create_document(
                            content,
                            "hierarchical_section",
                            {
                                "heading_level": level,
                                "heading_title": title,
                                "hierarchy_path": hierarchy_path,
                                "parent_headings": [item['title'] for item in hierarchy_stack[:-1]],
                                "depth": len(hierarchy_stack)
                            }
                        )
                        documents.append(doc)

        print(f"   → Created {len(documents)} hierarchical chunks")
        return documents

    def semantic_structural_chunking(self, text: str) -> List[Document]:
        """
        Example 5: Semantic + Structural hybrid
        Combines structural boundaries with semantic coherence
        """
        print("🧠 Example 5: Semantic-Structural Hybrid Chunking")

        documents = []

        # First, split by major structural elements
        sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            if re.match(r'^#{1,3}\s', section):
                heading_match = re.match(r'^(#{1,3})\s*(.+?)$', section, re.MULTILINE)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    content_lines = section.split('\n', 1)
                    content = content_lines[1] if len(content_lines) > 1 else ""

                    if content.strip():
                        # Split content into semantic chunks (by sentences/topics)
                        sentences = re.split(r'(?<=[.!?])\s+', content.strip())

                        # Group sentences into coherent chunks
                        chunk_size = 3  # sentences per chunk
                        for i in range(0, len(sentences), chunk_size):
                            chunk_sentences = sentences[i:i + chunk_size]
                            chunk_content = ' '.join(chunk_sentences)

                            if chunk_content.strip():
                                doc = self.create_document(
                                    chunk_content,
                                    "semantic_structural_chunk",
                                    {
                                        "heading_level": level,
                                        "heading_title": title,
                                        "sentence_count": len(chunk_sentences),
                                        "chunk_index": i // chunk_size,
                                        "is_under_heading": True
                                    }
                                )
                                documents.append(doc)

        print(f"   → Created {len(documents)} semantic-structural chunks")
        return documents

def demonstrate_chunking_examples():
    """Run all chunking examples with different text samples"""

    chunker = AdvancedStructuralChunker()

    # Example 1: Basic Headings
    print("\n" + "="*60)
    print("🧪 STRUCTURAL BLOCK CHUNKING EXAMPLES")
    print("="*60)

    example1_text = """
# Introduction
Welcome to our comprehensive guide on machine learning.

## What is ML?
Machine learning is a subset of artificial intelligence.
It enables computers to learn without being explicitly programmed.

## Applications
ML has numerous applications in various fields.
From image recognition to natural language processing.
"""

    chunks1 = chunker.split_markdown_headings(example1_text)
    print(f"📊 Total chunks created: {len(chunks1)}")

    # Example 2: Code Blocks
    print("\n" + "-"*50)
    example2_text = """
Here's how to implement a simple function:

```python
def hello_world():
    print("Hello, World!")
    return True
```

This function prints a greeting message.

```javascript
function helloWorld() {
    console.log("Hello, World!");
    return true;
}
```

Both functions achieve the same result in different languages.
"""

    chunks2 = chunker.split_code_blocks(example2_text)
    print(f"📊 Total chunks created: {len(chunks2)}")

    # Example 3: Lists and Tables
    print("\n" + "-"*50)
    example3_text = """
## Features

- Automatic text processing
- Real-time analysis
- Multi-language support

## Steps to Get Started

1. Install the package
2. Configure your API keys
3. Run the application

## Comparison Table

| Feature | Basic | Pro | Enterprise |
|---------|-------|-----|------------|
| Users | 1 | 10 | Unlimited |
| Storage | 1GB | 10GB | 100GB |
| Support | Email | Chat | Phone |
"""

    chunks3 = chunker.split_lists_and_tables(example3_text)
    print(f"📊 Total chunks created: {len(chunks3)}")

    # Example 4: Hierarchical
    print("\n" + "-"*50)
    example4_text = """
# Machine Learning Guide

## Fundamentals
### Supervised Learning
Supervised learning uses labeled data to train models.

### Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data.

## Advanced Topics
### Deep Learning
Deep learning uses neural networks with multiple layers.

#### Neural Networks
Neural networks are inspired by biological brains.

#### Convolutional Nets
CNNs excel at image processing tasks.
"""

    chunks4 = chunker.hierarchical_chunking(example4_text)
    print(f"📊 Total chunks created: {len(chunks4)}")

    # Example 5: Semantic-Structural Hybrid
    print("\n" + "-"*50)
    example5_text = """
# Data Science Workflow

## Data Collection
Data collection is the first step in any data science project. You need to gather relevant data from various sources. This includes databases, APIs, and external files. The quality of your data collection directly impacts your final results.

## Data Cleaning
Once you have collected the data, you need to clean it. This involves handling missing values, removing duplicates, and correcting inconsistencies. Data cleaning can take up to 80% of a data scientist's time. It's crucial for accurate analysis.

## Exploratory Analysis
Exploratory data analysis helps you understand your data. You create visualizations and calculate statistics. This step reveals patterns and insights. It guides your modeling decisions and helps you ask the right questions.
"""

    chunks5 = chunker.semantic_structural_chunking(example5_text)
    print(f"📊 Total chunks created: {len(chunks5)}")

    # Summary
    print("\n" + "="*60)
    print("📈 SUMMARY OF ALL EXAMPLES")
    print("="*60)
    print(f"Example 1 (Headings): {len(chunks1)} chunks")
    print(f"Example 2 (Code): {len(chunks2)} chunks")
    print(f"Example 3 (Lists/Tables): {len(chunks3)} chunks")
    print(f"Example 4 (Hierarchical): {len(chunks4)} chunks")
    print(f"Example 5 (Semantic-Structural): {len(chunks5)} chunks")

    total_chunks = len(chunks1) + len(chunks2) + len(chunks3) + len(chunks4) + len(chunks5)
    print(f"\n🎯 Total chunks across all examples: {total_chunks}")

    return {
        'example1': chunks1,
        'example2': chunks2,
        'example3': chunks3,
        'example4': chunks4,
        'example5': chunks5
    }

def analyze_chunk_metadata(chunks_dict):
    """Analyze and display metadata from chunks"""
    print("\n" + "="*60)
    print("🔍 CHUNK METADATA ANALYSIS")
    print("="*60)

    for example_name, chunks in chunks_dict.items():
        print(f"\n📋 {example_name.upper()} ANALYSIS:")

        block_types = {}
        total_words = 0
        total_chars = 0

        for chunk in chunks:
            block_type = chunk.metadata.get('block_type', 'unknown')
            block_types[block_type] = block_types.get(block_type, 0) + 1

            total_words += chunk.metadata.get('word_count', 0)
            total_chars += chunk.metadata.get('content_length', 0)

        print(f"   Block types: {block_types}")
        print(f"   Total words: {total_words}")
        print(f"   Total characters: {total_chars}")
        print(".2f")

def main():
    """Main function to run all examples"""
    print("🚀 Deep Structural Block Chunking Exploration")
    print("This tutorial demonstrates advanced chunking techniques one by one.")

    # Run all examples
    chunks_dict = demonstrate_chunking_examples()

    # Analyze results
    analyze_chunk_metadata(chunks_dict)

    print("\n🎉 Exploration completed! Each example shows different aspects of structural chunking.")
    print("\n💡 Key Takeaways:")
    print("   • Structural chunking preserves document hierarchy")
    print("   • Different content types need different chunking strategies")
    print("   • Rich metadata enables better retrieval and filtering")
    print("   • Hybrid approaches combine structural and semantic methods")

if __name__ == "__main__":
    main()
