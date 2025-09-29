import pdfplumber
import nltk
import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Download NLTK sentence tokenizer if not already present
nltk.download('punkt')

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def sliding_window_chunking(text: str, window_size: int = 100, stride: int = 50) -> List[str]:
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    chunks = []
    for start in range(0, len(words), stride):
        end = start + window_size
        chunk = words[start:end]
        if chunk:
            chunks.append(" ".join(chunk))
        if end >= len(words):
            break
    return chunks

def setup_chromadb_collection():
    """Setup ChromaDB client and collection for sliding window chunks"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="sliding_window_chunks",
        metadata={"description": "Collection for sliding window PDF chunking"}
    )
    return client, collection

def store_chunks_in_chromadb(chunks: List[str], collection, pdf_path: str, window_size: int, stride: int):
    """Store sliding window chunks in ChromaDB with embeddings"""
    # Initialize embeddings
    embedding_function = OpenAIEmbeddings()

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{idx}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "chunk_idx": idx,
            "pdf_source": pdf_path,
            "chunk_type": "sliding_window",
            "window_size": window_size,
            "stride": stride,
            "word_count": len(chunk.split())
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

    print(f"✅ Stored {len(chunks)} sliding window chunks in ChromaDB")

def create_langchain_documents(chunks: List[str], window_size: int, stride: int) -> List[Document]:
    """Convert sliding window chunks to LangChain Document objects"""
    documents = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_idx": idx,
                "chunk_type": "sliding_window",
                "window_size": window_size,
                "stride": stride,
                "word_count": len(chunk.split())
            }
        )
        documents.append(doc)
    return documents

def store_chunks_in_langchain_chroma(chunks: List[str], window_size: int, stride: int) -> Chroma:
    """Store sliding window chunks in LangChain Chroma vectorstore"""
    # Convert to LangChain documents
    documents = create_langchain_documents(chunks, window_size, stride)

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db_langchain",
        collection_name="sliding_window_langchain"
    )

    print(f"✅ Stored {len(documents)} sliding window chunks in LangChain Chroma")
    return vectorstore

def rag_query_langchain_sliding(vectorstore: Chroma, query: str) -> str:
    """LangChain RetrievalQA for sliding window chunks"""
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Create custom prompt for resume analysis with sliding windows
    PROMPT = PromptTemplate(
        template="""You are an expert assistant analyzing a resume/CV document.
Use the following retrieved text chunks from the document to answer the user's question.

Retrieved Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Reference specific parts of the resume when relevant
- Note that chunks may overlap, so look for consistent information across chunks
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and relevant to the resume content

Answer:""",
        input_variables=["context", "question"]
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]

def query_sliding_window_chunks(query: str, collection, top_k: int = 5):
    """Query the sliding window chunks collection"""
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

def rag_query_sliding_window(query: str, collection, top_k: int = 5):
    """Retrieval-Augmented Generation using sliding window chunks"""
    # Retrieve relevant chunks
    results = query_sliding_window_chunks(query, collection, top_k)

    if not results or not results.get('documents') or not results['documents']:
        return "No relevant information found in the document."

    # Extract retrieved content
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results.get('metadatas', [[]])
    metadatas = metadatas[0] if metadatas and len(metadatas) > 0 else []

    # Build context from retrieved chunks
    context_parts = []
    for doc, metadata in zip(docs, metadatas):
        chunk_idx = metadata.get('chunk_idx', 'unknown')
        context_parts.append(f"[Chunk {chunk_idx}] {doc}")

    context = "\n\n".join(context_parts)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Create prompt for RAG
    prompt = f"""You are an expert assistant analyzing a resume/CV document.
Use the following retrieved text chunks from the document to answer the user's question.

Retrieved Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Reference specific parts of the resume when relevant
- Note that chunks may overlap, so look for consistent information
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and relevant to the resume content

Answer:"""

    # Generate answer
    response = llm.invoke(prompt)
    return response.content

    # Generate answer
    response = llm.invoke(prompt)
    return response.content

def demonstrate_sliding_window_rag(collection):
    """Demonstrate sliding window RAG with sample queries"""
    print("\n🚀 Testing Sliding Window RAG")
    print("=" * 50)

    # Sample queries for resume analysis
    queries = [
        "What are the main skills mentioned?",
        "What experience does the candidate have?",
        "What projects has the candidate worked on?",
        "What is the candidate's educational background?"
    ]

    for query in queries:
        print(f"\n[yellow]Query:[/yellow] {query}")
        print("[blue]Generating answer...[/blue]")

        try:
            answer = rag_query_sliding_window(query, collection, top_k=5)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")

        print("-" * 50)

def demonstrate_langchain_sliding_window_rag(vectorstore: Chroma):
    """Demonstrate sliding window RAG using LangChain RetrievalQA"""
    print("\n🧪 Testing Sliding Window RAG with LangChain")
    print("=" * 50)

    queries = [
        "What are the main skills mentioned?",
        "What experience does the candidate have?",
        "What projects has the candidate worked on?",
        "What is the candidate's educational background?"
    ]

    for query in queries:
        print(f"\n[yellow]Query:[/yellow] {query}")
        print("[blue]Generating answer...[/blue]")

        try:
            answer = rag_query_langchain_sliding(vectorstore, query)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")

        print("-" * 50)

if __name__ == "__main__":
    print("[bold green]🔄 Sliding Window PDF Chunking with LangChain[/bold green]")
    print("=" * 60)

    pdf_path = "/Users/jyotiprakash/Desktop/python/rag-simulator/study-research/sliding-window/Resume JPM.pdf"

    # Extract text from PDF
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    total_words = len(nltk.word_tokenize(text))
    print(f"✅ Extracted {total_words} words")

    # Create sliding window chunks
    window_size = 100  # Number of words per chunk
    stride = 50        # Overlap between chunks
    print(f"🔄 Creating sliding window chunks (size={window_size}, stride={stride})...")
    chunks = sliding_window_chunking(text, window_size, stride)
    print(f"✅ Created {len(chunks)} chunks")

    # Display sample chunks
    print("\n📋 Sample Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  Chunk {i}: {chunk[:150]}...")

    # Setup traditional ChromaDB
    print("\n🗄️ Setting up ChromaDB...")
    client, collection = setup_chromadb_collection()

    # Store chunks in ChromaDB
    print("💾 Storing chunks in ChromaDB...")
    store_chunks_in_chromadb(chunks, collection, pdf_path, window_size, stride)

    # Demonstrate traditional RAG
    demonstrate_sliding_window_rag(collection)

    # Setup LangChain Chroma
    print("\n🔗 Setting up LangChain Chroma...")
    vectorstore = store_chunks_in_langchain_chroma(chunks, window_size, stride)

    # Demonstrate LangChain RAG
    demonstrate_langchain_sliding_window_rag(vectorstore)

    print("\n[bold green]🎉 Sliding window chunking and RAG demo completed![/bold green]")
