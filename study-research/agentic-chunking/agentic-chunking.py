import pdfplumber
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def agentic_chunking(text: str) -> list:
    """
    Agentic chunking implementation - splits text into actionable chunks
    based on semantic units like tasks, steps, and logical sections
    """
    # For this lab test, we'll implement a simple agentic chunking approach
    # that identifies sections, tasks, and actionable content

    import re

    chunks = []

    # Split by major sections (assuming resume format)
    sections = re.split(r'\n\s*(?=SKILLS|EXPERIENCE|PROJECTS|EDUCATION|CERTIFICATES)', text)

    for section in sections:
        if not section.strip():
            continue

        # Within each section, split by logical units
        # Look for bullet points, numbered items, or paragraph breaks
        if '●' in section or '•' in section:
            # Split by bullet points for task-oriented chunks
            bullets = re.split(r'\n\s*(?=●|•)', section)
            for bullet in bullets:
                if bullet.strip():
                    chunks.append(f"Task/Action: {bullet.strip()}")
        else:
            # For other sections, create semantic chunks
            paragraphs = section.split('\n\n')
            for para in paragraphs:
                if para.strip() and len(para.split()) > 5:  # Only meaningful paragraphs
                    chunks.append(f"Information: {para.strip()}")

    # If no chunks were created, fall back to sentence-based splitting
    if not chunks:
        import nltk
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        chunks = [f"Statement: {sentence}" for sentence in sentences if len(sentence.split()) > 3]

    return chunks

def setup_chromadb_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="agentic_chunks",
        metadata={"description": "Agentic chunking test collection"}
    )
    return client, collection

def store_agentic_chunks_in_chromadb(chunks: list, collection, pdf_path: str):
    embedding_function = OpenAIEmbeddings()
    ids = []
    documents = []
    metadatas = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"agentic_{idx}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "chunk_idx": idx,
            "pdf_source": pdf_path,
            "chunk_type": "agentic"
        })
    embeddings_list = embedding_function.embed_documents(documents)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas
    )
    print(f"✅ Stored {len(chunks)} agentic chunks in ChromaDB")

def query_agentic_chunks(query: str, collection, top_k: int = 5):
    embedding_function = OpenAIEmbeddings()
    query_embedding = embedding_function.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def rag_query_agentic(query: str, collection, top_k: int = 5):
    results = query_agentic_chunks(query, collection, top_k)
    if not results or not results.get('documents') or not results['documents']:
        return "No relevant information found in the document."
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results.get('metadatas', [[]])
    metadatas = metadatas[0] if metadatas and len(metadatas) > 0 else []
    context_parts = []
    for doc, metadata in zip(docs, metadatas):
        chunk_idx = metadata.get('chunk_idx', 'unknown')
        context_parts.append(f"[Agentic Chunk {chunk_idx}] {doc}")
    context = "\n\n".join(context_parts)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    prompt = f"""You are an expert assistant analyzing a technical document.\nUse the following retrieved agentic chunks to answer the user's question.\n\nRetrieved Context:\n{context}\n\nQuestion: {query}\n\nInstructions:\n- Provide a clear, accurate answer based on the retrieved information\n- Reference specific steps, actions, or code when relevant\n- If the context doesn't contain enough information, say so clearly\n- Keep your answer focused and relevant\n\nAnswer:"""
    response = llm.invoke(prompt)
    return response.content

def demonstrate_agentic_chunking_rag(collection):
    print("\n🧪 Testing Agentic Chunking RAG")
    print("=" * 50)
    queries = [
        "Who is jyoti prakash?",
        "How many experiences does he have?",
        "is jyoti prakash working currently?",
        "What are his key skills?",
        "List the projects mentioned in the resume.",
        "What technologies does he specialize in?",
        "Describe his role at SettleMint.",
        "What certifications does he hold?",
        "What is his educational background?",
        "What programming languages is he proficient in?",
        "What frameworks does he have experience with?",
        "What tools does he use for development?",
    ]
    for query in queries:
        print(f"\n[yellow]Query:[/yellow] {query}")
        print("[blue]Generating answer...[/blue]")
        try:
            answer = rag_query_agentic(query, collection, top_k=5)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")
        print("-" * 50)

if __name__ == "__main__":
    print("[bold green]🧪 Agentic Chunking Lab Test with ChromaDB[/bold green]")
    print("=" * 60)
    pdf_path = "../sliding-window/Resume JPM.pdf"  # Correct path to the PDF
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✅ Extracted {len(text.split())} words")
    print("🔗 Performing agentic chunking...")
    chunks = agentic_chunking(text)
    print(f"✅ Created {len(chunks)} agentic chunks")
    print("\n📋 Sample Agentic Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  Agentic Chunk {i}: {chunk[:150]}...")
    print("\n🗄️ Setting up ChromaDB...")
    client, collection = setup_chromadb_collection()
    print("💾 Storing agentic chunks in ChromaDB...")
    store_agentic_chunks_in_chromadb(chunks, collection, pdf_path)
    demonstrate_agentic_chunking_rag(collection)
    print("\n[bold green]🎉 Agentic chunking lab test completed![/bold green]")
