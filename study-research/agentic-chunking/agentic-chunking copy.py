import pdfplumber
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from dotenv import load_dotenv
from typing import List

load_dotenv()

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def agentic_chunking(text: str) -> List[Document]:
    """
    Enhanced agentic chunking using LangChain text splitters
    Creates actionable chunks optimized for agent workflows
    """
    import re

    # First, split by major semantic sections
    sections = re.split(r'\n\s*(?=SKILLS|EXPERIENCE|PROJECTS|EDUCATION|CERTIFICATES)', text)

    documents = []

    for section_idx, section in enumerate(sections):
        if not section.strip():
            continue

        section_name = "General"
        if "SKILLS" in section:
            section_name = "Skills"
        elif "EXPERIENCE" in section:
            section_name = "Experience"
        elif "PROJECTS" in section:
            section_name = "Projects"
        elif "EDUCATION" in section:
            section_name = "Education"
        elif "CERTIFICATES" in section:
            section_name = "Certificates"

        # Use LangChain RecursiveCharacterTextSplitter for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Larger chunks for agent reasoning
            chunk_overlap=100,  # Overlap for context preservation
            separators=["\n●", "\n•", "\n\n", ". ", " ", ""]  # Prioritize bullet points and paragraphs
        )

        section_chunks = text_splitter.split_text(section.strip())

        for chunk_idx, chunk in enumerate(section_chunks):
            if chunk.strip():
                # Create agent-friendly metadata
                metadata = {
                    "section": section_name,
                    "chunk_type": "agentic",
                    "chunk_idx": chunk_idx,
                    "section_idx": section_idx,
                    "word_count": len(chunk.split()),
                    "actionable": "task" if any(keyword in chunk.lower() for keyword in
                                               ["developed", "built", "created", "worked", "led", "managed"]) else "information"
                }

                doc = Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
                documents.append(doc)

    return documents

def setup_chromadb_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="agentic_chunks",
        metadata={"description": "Agentic chunking test collection"}
    )
    return client, collection

def store_agentic_chunks_in_chromadb(documents: List[Document], collection_name: str = "agentic_chunks_langchain"):
    """Store agentic chunks in ChromaDB using LangChain's Chroma vectorstore"""
    # Create Chroma vectorstore from documents
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

    print(f"✅ Stored {len(documents)} agentic chunks in ChromaDB")
    return vectorstore

def rag_query_langchain(vectorstore, query: str) -> str:
    """Retrieval-Augmented Generation using LangChain's RetrievalQA"""
    # Create a custom prompt for agentic chunking
    prompt_template = """You are an expert assistant analyzing a technical resume/CV document.
Use the following retrieved context chunks to answer the user's question.

Context: {context}

Question: {question}

Instructions:
- Focus on actionable information and agent-relevant details
- Reference specific sections, skills, or experiences when relevant
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and relevant to agent workflows

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create RetrievalQA chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]

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

def demonstrate_agentic_chunking_rag(vectorstore):
    """Demonstrate agentic chunking RAG using LangChain"""
    print("\n🧪 Testing Agentic Chunking RAG with LangChain")
    print("=" * 50)

    queries = [
        "Who is jyoti prakash?",
        "What are his key skills?",
        "What projects has he worked on?",
        "What is his current role?"
    ]

    for query in queries:
        print(f"\n[yellow]Query:[/yellow] {query}")
        print("[blue]Generating answer...[/blue]")

        try:
            answer = rag_query_langchain(vectorstore, query)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")

        print("-" * 50)

if __name__ == "__main__":
    print("[bold green]🧪 Agentic Chunking Lab Test with ChromaDB[/bold green]")
    print("=" * 60)
    pdf_path = "/Users/jyotiprakash/Desktop/python/rag-simulator/study-research/sliding-window/Resume JPM.pdf"  # Absolute path to the PDF
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✅ Extracted {len(text.split())} words")
    print("🔗 Performing agentic chunking...")
    chunks = agentic_chunking(text)
    print(f"✅ Created {len(chunks)} agentic chunks")
    print("\n📋 Sample Agentic Chunks:")
    for i, doc in enumerate(chunks[:3], 1):
        print(f"  Agentic Chunk {i}: {doc.page_content[:150]}...")
        print(f"    Metadata: {doc.metadata}")
    print("\n🗄️ Setting up ChromaDB...")
    client, collection = setup_chromadb_collection()
    print("💾 Storing agentic chunks in ChromaDB...")
    vectorstore = store_agentic_chunks_in_chromadb(chunks)
    demonstrate_agentic_chunking_rag(vectorstore)
    print("\n[bold green]🎉 Agentic chunking lab test completed![/bold green]")
