from langchain.text_splitter import PythonCodeTextSplitter
# Structural Block Chunking for Algorithm PDF
# This script extracts structured blocks (title, sections, code, examples) from algorithms_examples.pdf

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from rich.console import Console
from rich.tree import Tree
import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

console = Console()

@dataclass
class Block:
	block_type: str
	content: str
	page_number: int
	metadata: Dict[str, Any]

def setup_chromadb_collection():
	"""Setup ChromaDB client and collection for algorithm blocks"""
	client = chromadb.PersistentClient(path="./chroma_db")
	collection = client.get_or_create_collection(
		name="algorithm_blocks",
		metadata={"description": "Collection for algorithm PDF structural blocks"}
	)
	return client, collection

def store_blocks_in_chromadb(blocks: List[Block], collection):
	"""Store blocks in ChromaDB with embeddings"""
	# Initialize embeddings
	embedding_function = OpenAIEmbeddings()

	# Prepare data for ChromaDB
	ids = []
	documents = []
	metadatas = []

	for block in blocks:
		block_id = f"{block.block_type}_{block.page_number}_{block.metadata.get('section_num', '0')}_{block.metadata.get('chunk_idx', '0')}"
		ids.append(block_id)
		documents.append(block.content)
		metadatas.append({
			"block_type": block.block_type,
			"page_number": block.page_number,
			**block.metadata
		})

	# Generate embeddings
	embeddings_list = embedding_function.embed_documents(documents)

	# Store in ChromaDB
	collection.add(
		ids=ids,
		documents=documents,
		embeddings=embeddings_list,
		metadatas=metadatas
	)

	console.print(f"[green]✅ Stored {len(blocks)} blocks in ChromaDB[/green]")

def query_blocks(query: str, collection, top_k: int = 3):
	"""Query the algorithm blocks collection"""
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

def rag_query(query: str, collection, top_k: int = 3):
	"""Retrieval-Augmented Generation: Query blocks and generate answer using LLM"""
	# Retrieve relevant blocks
	results = query_blocks(query, collection, top_k)

	if not results or not results.get('documents') or not results['documents']:
		return "No relevant information found in the algorithm documentation."

	# Extract retrieved content
	docs = results['documents'][0] if results['documents'] else []
	metadatas = results.get('metadatas', [[]])
	metadatas = metadatas[0] if metadatas and len(metadatas) > 0 else []

	# Build context from retrieved blocks
	context_parts = []
	for doc, metadata in zip(docs, metadatas):
		block_type = metadata.get('block_type', 'unknown')
		section_num = metadata.get('section_num', '')
		context_parts.append(f"[{block_type.upper()}] {doc}")

	context = "\n\n".join(context_parts)

	# Initialize LLM
	llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

	# Create prompt for RAG
	prompt = f"""You are an expert programming assistant specializing in algorithms and data structures.
Use the following retrieved information from an algorithms reference document to answer the user's question.

Retrieved Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Include relevant code examples when available
- Explain concepts step-by-step when appropriate
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and relevant

Answer:"""

	# Generate answer
	response = llm.invoke(prompt)
	return response.content

def extract_blocks_from_pdf(pdf_path: str) -> List[Block]:
	blocks = []
	with pdfplumber.open(pdf_path) as pdf:
		for page_num, page in enumerate(pdf.pages, 1):
			text = page.extract_text()
			if not text:
				continue
			lines = text.splitlines()
			i = 0
			# Title detection (first page, first line)
			if page_num == 1 and lines:
				title = lines[0].strip()
				blocks.append(Block(
					block_type="title",
					content=title,
					page_number=page_num,
					metadata={"line": 0}
				))
				i = 1
			# Section/algorithm detection
			section_pattern = re.compile(r"^(\d+)\.\s*(.+)")
			while i < len(lines):
				line = lines[i].strip()
				section_match = section_pattern.match(line)
				if section_match:
					section_num = section_match.group(1)
					section_title = section_match.group(2)
					blocks.append(Block(
						block_type="section_heading",
						content=section_title,
						page_number=page_num,
						metadata={"section_num": section_num, "line": i}
					))
					i += 1
					# Description: lines until code block (starts with 'def ' or 'import ')
					desc_lines = []
					while i < len(lines) and not (lines[i].strip().startswith("def ") or lines[i].strip().startswith("import ")):
						if lines[i].strip():
							desc_lines.append(lines[i].strip())
						i += 1
					if desc_lines:
						blocks.append(Block(
							block_type="description",
							content=" ".join(desc_lines),
							page_number=page_num,
							metadata={"section_num": section_num}
						))
					# Code block: lines until example or next section
					code_lines = []
					while i < len(lines) and (lines[i].strip().startswith("def ") or lines[i].strip().startswith("import ") or lines[i].strip().startswith("while ") or lines[i].strip().startswith("if ") or lines[i].strip().startswith("elif ") or lines[i].strip().startswith("else:") or lines[i].strip().startswith("for ") or lines[i].strip().startswith("return") or lines[i].strip().startswith("distance =") or lines[i].strip().startswith("pq =") or lines[i].strip().startswith("current_distance") or lines[i].strip().startswith("current_node") or lines[i].strip().startswith("distances =") or lines[i].strip().startswith("arr =") or lines[i].strip().startswith("graph =") or lines[i].strip().startswith("print(") or lines[i].strip().startswith("result =") or lines[i].strip().startswith("i =") or lines[i].strip().startswith("j =") or lines[i].strip().startswith("result.extend") or lines[i].strip().startswith("return result")):
						code_lines.append(lines[i])
						i += 1
					if code_lines:
						# Use LangChain PythonCodeTextSplitter for granular chunking
						code_text = "\n".join(code_lines)
						splitter = PythonCodeTextSplitter()
						code_chunks = splitter.split_text(code_text)
						for idx, chunk in enumerate(code_chunks):
							blocks.append(Block(
								block_type="code_chunk",
								content=chunk,
								page_number=page_num,
								metadata={"section_num": section_num, "chunk_idx": idx}
							))
					# Example: lines starting with '# Example' and following lines
					if i < len(lines) and lines[i].strip().startswith("# Example"):
						example_lines = [lines[i].strip()]
						i += 1
						while i < len(lines) and lines[i].strip():
							example_lines.append(lines[i].strip())
							i += 1
						blocks.append(Block(
							block_type="example",
							content="\n".join(example_lines),
							page_number=page_num,
							metadata={"section_num": section_num}
						))
				else:
					i += 1
	return blocks

def display_block_hierarchy(blocks: List[Block]):
	tree = Tree("[bold blue]PDF Block Structure[/bold blue]")
	title_block = next((b for b in blocks if b.block_type == "title"), None)
	if title_block:
		title_node = tree.add(f"[bold magenta]Title:[/bold magenta] {title_block.content}")
	else:
		title_node = tree
	# Group by section
	section_blocks = [b for b in blocks if b.block_type == "section_heading"]
	for section in section_blocks:
		section_node = title_node.add(f"[bold green]Section:[/bold green] {section.content}")
		# Add description, code, example
		desc = next((b for b in blocks if b.block_type == "description" and b.metadata.get("section_num") == section.metadata.get("section_num")), None)
		if desc:
			section_node.add(f"[yellow]Description:[/yellow] {desc.content}")
		# Look for code chunks
		code_chunks = [b for b in blocks if b.block_type == "code_chunk" and b.metadata.get("section_num") == section.metadata.get("section_num")]
		if code_chunks:
			code_node = section_node.add("[cyan]Code Chunks:[/cyan]")
			for chunk in code_chunks:
				code_node.add(f"[blue]Chunk {chunk.metadata.get('chunk_idx', 0)}:[/blue]\n{chunk.content}")
		example = next((b for b in blocks if b.block_type == "example" and b.metadata.get("section_num") == section.metadata.get("section_num")), None)
		if example:
			section_node.add(f"[white]Example:[/white]\n{example.content}")
	console.print(tree)

def demonstrate_retrieval(collection):
	"""Demonstrate RAG (Retrieval-Augmented Generation) with sample queries"""
	console.print("\n[bold blue]🤖 Testing RAG with LLM Answers[/bold blue]")
	console.print("=" * 50)

	# Sample queries for algorithm retrieval
	queries = [
		"How does binary search work?",
		"Explain Dijkstra's algorithm",
		"Show me merge sort implementation",
		"What is the time complexity of these algorithms?"
	]

	for query in queries:
		console.print(f"\n[yellow]Query:[/yellow] {query}")
		console.print("[blue]Generating answer...[/blue]")

		try:
			answer = rag_query(query, collection, top_k=3)
			console.print(f"[green]Answer:[/green]\n{answer}")
		except Exception as e:
			console.print(f"[red]Error generating answer: {e}[/red]")

		console.print("-" * 50)

if __name__ == "__main__":
	console.print("[bold green]🚀 Algorithm PDF Structural Chunking with ChromaDB[/bold green]")
	console.print("=" * 60)

	pdf_path = "algorithms_examples.pdf"

	# Extract blocks from PDF
	console.print("📄 Extracting structural blocks from PDF...")
	blocks = extract_blocks_from_pdf(pdf_path)
	console.print(f"✅ Extracted {len(blocks)} blocks")

	# Display block hierarchy
	display_block_hierarchy(blocks)

	# Setup ChromaDB
	console.print("\n🗄️ Setting up ChromaDB...")
	client, collection = setup_chromadb_collection()

	# Store blocks in ChromaDB
	console.print("💾 Storing blocks in ChromaDB...")
	store_blocks_in_chromadb(blocks, collection)

	# Demonstrate retrieval
	demonstrate_retrieval(collection)

	console.print("\n[bold green]🎉 Complete workflow demonstration finished![/bold green]")
