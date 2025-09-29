"""
LangChain Structural Block Chunking - Complete Guide
=====================================================

This file explores ALL text splitting functions provided by LangChain,
with a focus on structural and document-based chunking strategies.
"""

import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List
import re
from rich import print
from dotenv import load_dotenv

# Import all LangChain text splitters
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter,
    Language,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter
)

load_dotenv()

class LangChainSplitterExplorer:
    """
    Comprehensive explorer of all LangChain text splitting methods
    """

    def __init__(self):
        self.sample_texts = {
            'markdown': self._get_markdown_sample(),
            'python_code': self._get_python_sample(),
            'html': self._get_html_sample(),
            'json': self._get_json_sample(),
            'latex': self._get_latex_sample(),
            'mixed': self._get_mixed_sample()
        }

    def _get_markdown_sample(self):
        return """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled training data to learn a mapping from inputs to outputs.

### Classification
Classification is a type of supervised learning where the goal is to predict categorical labels.

### Regression
Regression predicts continuous numerical values.

## Unsupervised Learning

Unsupervised learning finds patterns in data without labeled examples.

### Clustering
Clustering groups similar data points together.

### Dimensionality Reduction
Reduces the number of features while preserving important information.

## Deep Learning

Deep learning uses neural networks with multiple layers.

### Neural Networks
- Feedforward networks
- Convolutional networks
- Recurrent networks

### Applications
1. Computer vision
2. Natural language processing
3. Speech recognition
"""

    def _get_python_sample(self):
        return """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Model parameters
    input_size = 784
    hidden_size = 128
    output_size = 10

    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Neural network training example")
"""

    def _get_html_sample(self):
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Webpage</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background-color: #f0f0f0; padding: 20px; }
        .content { margin: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Welcome to Our Website</h1>
        <p>This is a sample webpage demonstrating HTML structure.</p>
    </div>

    <div class="content">
        <h2>About Us</h2>
        <p>We are a company that specializes in web development and data science.</p>

        <h3>Our Services</h3>
        <ul>
            <li>Web Development</li>
            <li>Data Analysis</li>
            <li>Machine Learning</li>
        </ul>

        <h3>Contact Information</h3>
        <p>Email: info@company.com</p>
        <p>Phone: (123) 456-7890</p>
    </div>
</body>
</html>
"""

    def _get_json_sample(self):
        return """
{
  "company": {
    "name": "TechCorp",
    "founded": 2010,
    "employees": [
      {
        "name": "Alice Johnson",
        "role": "CEO",
        "department": "Executive",
        "skills": ["leadership", "strategy", "finance"]
      },
      {
        "name": "Bob Smith",
        "role": "CTO",
        "department": "Engineering",
        "skills": ["software development", "architecture", "python"]
      },
      {
        "name": "Carol Davis",
        "role": "Data Scientist",
        "department": "Analytics",
        "skills": ["machine learning", "statistics", "python", "r"]
      }
    ],
    "departments": {
      "engineering": {
        "headcount": 25,
        "technologies": ["python", "javascript", "react", "docker"]
      },
      "analytics": {
        "headcount": 10,
        "technologies": ["python", "r", "tensorflow", "spark"]
      }
    }
  }
}
"""

    def _get_latex_sample(self):
        return """
\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{graphicx}

\\title{Mathematical Foundations of Machine Learning}
\\author{Dr. Jane Smith}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}

Machine learning has become a cornerstone of modern computational methods. This paper explores the mathematical foundations that underpin these techniques.

\\subsection{Linear Algebra}

Linear algebra provides the mathematical framework for understanding data transformations and neural network operations.

\\begin{equation}
\\mathbf{y} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}
\\end{equation}

Where $\\mathbf{W}$ represents the weight matrix, $\\mathbf{x}$ is the input vector, and $\\mathbf{b}$ is the bias term.

\\subsection{Calculus}

Calculus is essential for optimization in machine learning algorithms.

\\begin{equation}
\\frac{\\partial L}{\\partial w} = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i) x_i
\\end{equation}

\\section{Applications}

\\subsection{Computer Vision}

Computer vision applications include:
\\begin{itemize}
\\item Image classification
\\item Object detection
\\item Image segmentation
\\end{itemize}

\\subsection{Natural Language Processing}

NLP applications include:
\\begin{enumerate}
\\item Text classification
\\item Machine translation
\\item Sentiment analysis
\\end{enumerate}

\\end{document}
"""

    def _get_mixed_sample(self):
        return """
# Project Documentation

## Overview
This project implements a comprehensive machine learning pipeline for text analysis.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```python
from text_analyzer import TextAnalyzer

analyzer = TextAnalyzer()
results = analyzer.analyze("Sample text for analysis")
print(results)
```

## API Reference

### Class: TextAnalyzer

#### Methods

- `analyze(text: str) -> dict`: Analyzes input text
- `preprocess(text: str) -> str`: Preprocesses text data
- `extract_features(text: str) -> list`: Extracts features from text

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text to analyze |
| model | str | Model name (default: 'default') |
| threshold | float | Confidence threshold (default: 0.5) |

## Configuration

The system can be configured using environment variables:

```bash
export MODEL_PATH=/path/to/model
export API_KEY=your_api_key_here
export DEBUG=true
```

## Examples

### Basic Usage

```python
analyzer = TextAnalyzer()
result = analyzer.analyze("Hello, world!")
print(f"Sentiment: {result['sentiment']}")
```

### Advanced Usage

```python
analyzer = TextAnalyzer(model='advanced')
results = analyzer.batch_analyze(texts)
for result in results:
    print(f"Text: {result['text'][:50]}...")
    print(f"Confidence: {result['confidence']:.2f}")
```
"""

    def explore_character_splitters(self):
        """Explore character-based text splitters"""
        print("\n" + "="*60)
        print("🔤 CHARACTER-BASED SPLITTERS")
        print("="*60)

        text = self.sample_texts['mixed']

        # 1. CharacterTextSplitter
        print("\n1️⃣ CharacterTextSplitter")
        splitter = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separator=""
        )
        chunks = splitter.create_documents([text])
        print(f"   → Created {len(chunks)} chunks")
        print(f"   → Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

        # 2. RecursiveCharacterTextSplitter
        print("\n2️⃣ RecursiveCharacterTextSplitter")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""]
        )
        chunks = splitter.create_documents([text])
        print(f"   → Created {len(chunks)} chunks")
        print(f"   → Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    def explore_language_splitters(self):
        """Explore language-specific text splitters"""
        print("\n" + "="*60)
        print("💻 LANGUAGE-SPECIFIC SPLITTERS")
        print("="*60)

        # 1. MarkdownTextSplitter
        print("\n1️⃣ MarkdownTextSplitter")
        splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.create_documents([self.sample_texts['markdown']])
        print(f"   → Created {len(chunks)} chunks from markdown")

        # 2. PythonCodeTextSplitter
        print("\n2️⃣ PythonCodeTextSplitter")
        splitter = PythonCodeTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.create_documents([self.sample_texts['python_code']])
        print(f"   → Created {len(chunks)} chunks from Python code")

        # 3. RecursiveCharacterTextSplitter for JavaScript
        print("\n3️⃣ RecursiveCharacterTextSplitter (JavaScript)")
        js_code = """
function processData(data) {
    const result = data.map(item => {
        return item.value * 2;
    });
    return result.filter(x => x > 10);
}

class DataProcessor {
    constructor(config) {
        this.config = config;
    }

    async process(input) {
        try {
            const output = await this.transform(input);
            return this.validate(output);
        } catch (error) {
            console.error('Processing failed:', error);
            throw error;
        }
    }
}
"""
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=150,
            chunk_overlap=30
        )
        chunks = splitter.create_documents([js_code])
        print(f"   → Created {len(chunks)} chunks from JavaScript code")

        # 4. LatexTextSplitter
        print("\n4️⃣ LatexTextSplitter")
        splitter = LatexTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = splitter.create_documents([self.sample_texts['latex']])
        print(f"   → Created {len(chunks)} chunks from LaTeX")

    def explore_structural_splitters(self):
        """Explore structural and document-based splitters"""
        print("\n" + "="*60)
        print("🏗️ STRUCTURAL & DOCUMENT SPLITTERS")
        print("="*60)

        # 1. HTMLHeaderTextSplitter
        print("\n1️⃣ HTMLHeaderTextSplitter")
        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
        )
        chunks = splitter.split_text(self.sample_texts['html'])
        print(f"   → Created {len(chunks)} chunks from HTML")

        # 2. RecursiveJsonSplitter
        print("\n2️⃣ RecursiveJsonSplitter")
        import json
        json_data = json.loads(self.sample_texts['json'])
        splitter = RecursiveJsonSplitter(max_chunk_size=200)
        chunks = splitter.create_documents([json_data])
        print(f"   → Created {len(chunks)} chunks from JSON")

    def explore_token_splitters(self):
        """Explore token-based text splitters"""
        print("\n" + "="*60)
        print("🎫 TOKEN-BASED SPLITTERS")
        print("="*60)

        text = self.sample_texts['mixed']

        # 1. TokenTextSplitter
        print("\n1️⃣ TokenTextSplitter")
        try:
            splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
            chunks = splitter.create_documents([text])
            print(f"   → Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   → Error: {e}")

        # 2. SentenceTransformersTokenTextSplitter
        print("\n2️⃣ SentenceTransformersTokenTextSplitter")
        try:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_size=100,
                chunk_overlap=20,
                model_name="all-MiniLM-L6-v2"
            )
            chunks = splitter.create_documents([text])
            print(f"   → Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   → Error: {e}")

    def explore_nlp_splitters(self):
        """Explore NLP-based text splitters"""
        print("\n" + "="*60)
        print("🧠 NLP-BASED SPLITTERS")
        print("="*60)

        text = self.sample_texts['markdown']

        # 1. NLTKTextSplitter
        print("\n1️⃣ NLTKTextSplitter")
        try:
            splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = splitter.create_documents([text])
            print(f"   → Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   → NLTK not available: {e}")

        # 2. SpacyTextSplitter
        print("\n2️⃣ SpacyTextSplitter")
        try:
            splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = splitter.create_documents([text])
            print(f"   → Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   → spaCy not available: {e}")

    def explore_language_agnostic_splitters(self):
        """Explore language-agnostic splitters using Language enum"""
        print("\n" + "="*60)
        print("🌍 LANGUAGE-AGNOSTIC SPLITTERS")
        print("="*60)

        # Demonstrate RecursiveCharacterTextSplitter with different languages
        languages = [
            ("Python", Language.PYTHON),
            ("JavaScript", Language.JS),
            ("TypeScript", Language.TS),
            ("Java", Language.JAVA),
            ("C++", Language.CPP),
            ("C#", Language.CSHARP),
            ("Go", Language.GO),
            ("Rust", Language.RUST),
            ("PHP", Language.PHP),
            ("Ruby", Language.RUBY),
            ("Swift", Language.SWIFT),
            ("Kotlin", Language.KOTLIN),
            ("Scala", Language.SCALA),
            ("HTML", Language.HTML),
            ("Markdown", Language.MARKDOWN),
            ("LaTeX", Language.LATEX),
        ]

        print(f"LangChain supports {len(languages)} programming languages and formats!")

        # Test a few examples
        test_cases = [
            ("Python", Language.PYTHON, self.sample_texts['python_code'][:300]),
            ("Markdown", Language.MARKDOWN, self.sample_texts['markdown'][:300]),
            ("HTML", Language.HTML, self.sample_texts['html'][:300]),
        ]

        for lang_name, lang_enum, sample_text in test_cases:
            print(f"\n🔧 {lang_name} Language Splitter")
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=150,
                    chunk_overlap=30
                )
                chunks = splitter.create_documents([sample_text])
                print(f"   → Created {len(chunks)} chunks")
            except Exception as e:
                print(f"   → Error: {e}")

    def compare_splitters_performance(self):
        """Compare performance of different splitters on the same text"""
        print("\n" + "="*60)
        print("⚡ SPLITTER PERFORMANCE COMPARISON")
        print("="*60)

        text = self.sample_texts['mixed']
        splitters = {
            "CharacterTextSplitter": CharacterTextSplitter(chunk_size=200, chunk_overlap=50),
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=50
            ),
            "MarkdownTextSplitter": MarkdownTextSplitter(chunk_size=200, chunk_overlap=50),
        }

        print(f"Input text length: {len(text)} characters")
        print(f"Input text words: {len(text.split())} words")

        results = []
        for name, splitter in splitters.items():
            try:
                chunks = splitter.create_documents([text])
                avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
                results.append({
                    'name': name,
                    'chunks': len(chunks),
                    'avg_size': avg_chunk_size,
                    'total_size': sum(len(c.page_content) for c in chunks)
                })
            except Exception as e:
                results.append({
                    'name': name,
                    'error': str(e)
                })

        # Display results
        print("\n📊 Performance Results:")
        print("-" * 50)
        for result in results:
            if 'error' in result:
                print(f"{result['name']}: ERROR - {result['error']}")
            else:
                print(f"{result['name']}:")
                print(".1f")
                print(f"  - Total size: {result['total_size']} chars")

    def demonstrate_advanced_usage(self):
        """Demonstrate advanced usage patterns"""
        print("\n" + "="*60)
        print("🚀 ADVANCED USAGE PATTERNS")
        print("="*60)

        # 1. Custom separators
        print("\n1️⃣ Custom Separators")
        text = """
        CHAPTER 1: Introduction
        This is the first chapter.

        CHAPTER 2: Background
        This covers background information.

        SECTION 2.1: Historical Context
        Historical details here.

        SECTION 2.2: Current State
        Current information here.
        """

        splitter = RecursiveCharacterTextSplitter(
            separators=["\nCHAPTER ", "\nSECTION ", "\n\n", "\n", " ", ""],
            chunk_size=100,
            chunk_overlap=20
        )
        chunks = splitter.create_documents([text])
        print(f"   → Created {len(chunks)} chunks with custom separators")

        # 2. Length function customization
        print("\n2️⃣ Custom Length Function")
        def token_length_function(text):
            # Simple tokenization by whitespace
            return len(text.split())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,  # tokens
            chunk_overlap=10,
            length_function=token_length_function
        )
        chunks = splitter.create_documents([text])
        print(f"   → Created {len(chunks)} chunks using token-based length")

        # 3. Metadata preservation
        print("\n3️⃣ Metadata Preservation")
        documents = [
            Document(page_content=text, metadata={"source": "book", "chapter": "introduction"}),
        ]

        splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)

        print(f"   → Split into {len(split_docs)} documents")
        for i, doc in enumerate(split_docs[:2]):
            print(f"   → Chunk {i+1} metadata: {doc.metadata}")

def main():
    """Main function to run all explorations"""
    print("🚀 LangChain Structural Block Chunking - Complete Exploration")
    print("This script demonstrates ALL text splitting functions in LangChain")

    explorer = LangChainSplitterExplorer()

    # Run all explorations
    explorer.explore_character_splitters()
    explorer.explore_language_splitters()
    explorer.explore_structural_splitters()
    explorer.explore_token_splitters()
    explorer.explore_nlp_splitters()
    explorer.explore_language_agnostic_splitters()
    explorer.compare_splitters_performance()
    explorer.demonstrate_advanced_usage()

    print("\n" + "="*60)
    print("🎉 COMPLETE LANGCHAIN TEXT SPLITTER EXPLORATION FINISHED!")
    print("="*60)
    print("\n📚 SUMMARY:")
    print("✅ Character-based splitters (2 types)")
    print("✅ Language-specific splitters (4+ types)")
    print("✅ Structural splitters (2 types)")
    print("✅ Token-based splitters (2 types)")
    print("✅ NLP-based splitters (2 types)")
    print("✅ Language-agnostic splitters (25+ languages)")
    print("✅ Advanced usage patterns")
    print("\n💡 Choose the right splitter based on your document type and use case!")

if __name__ == "__main__":
    main()
