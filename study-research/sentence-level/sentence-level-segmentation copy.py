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

def segment_pdf_sentences(pdf_path):
    sentences = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Use NLTK to split text into sentences
                page_sentences = nltk.sent_tokenize(text)
                sentences.extend(page_sentences)
    return sentences

def setup_chromadb_collection():
    """Setup ChromaDB client and collection for sentence-level chunks"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="sentence_level_chunks",
        metadata={"description": "Collection for sentence-level PDF segmentation"}
    )
    return client, collection

def store_sentences_in_chromadb(sentences: list, collection, pdf_path: str):
    """Store sentences in ChromaDB with embeddings"""
    # Initialize embeddings
    embedding_function = OpenAIEmbeddings()

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for idx, sentence in enumerate(sentences):
        sentence_id = f"sentence_{idx}"
        ids.append(sentence_id)
        documents.append(sentence)
        metadatas.append({
            "sentence_idx": idx,
            "pdf_source": pdf_path,
            "chunk_type": "sentence",
            "word_count": len(sentence.split())
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

    print(f"✅ Stored {len(sentences)} sentences in ChromaDB")

def create_langchain_documents(sentences: List[str]) -> List[Document]:
    """Convert sentences to LangChain Document objects"""
    documents = []
    for idx, sentence in enumerate(sentences):
        doc = Document(
            page_content=sentence,
            metadata={
                "sentence_idx": idx,
                "chunk_type": "sentence",
                "word_count": len(sentence.split())
            }
        )
        documents.append(doc)
    return documents

def store_sentences_in_langchain_chroma(sentences: List[str]) -> Chroma:
    """Store sentences in LangChain Chroma vectorstore"""
    # Convert to LangChain documents
    documents = create_langchain_documents(sentences)

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db_langchain",
        collection_name="sentence_level_langchain"
    )

    print(f"✅ Stored {len(documents)} sentences in LangChain Chroma")
    return vectorstore

def rag_query_langchain(vectorstore: Chroma, query: str) -> str:
    """LangChain RetrievalQA for sentence-level chunks"""
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Create custom prompt for resume analysis
    PROMPT = PromptTemplate(
        template="""You are an expert assistant analyzing a resume/CV document.
Use the following retrieved sentences from the document to answer the user's question.

Retrieved Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Reference specific parts of the resume when relevant
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

def query_sentences(query: str, collection, top_k: int = 5):
    """Query the sentence-level collection"""
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

def rag_query_sentences(query: str, collection, top_k: int = 5):
    """Retrieval-Augmented Generation using sentence-level chunks"""
    # Retrieve relevant sentences
    results = query_sentences(query, collection, top_k)

    if not results or not results.get('documents') or not results['documents']:
        return "No relevant information found in the document."

    # Extract retrieved content
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results.get('metadatas', [[]])
    metadatas = metadatas[0] if metadatas and len(metadatas) > 0 else []

    # Build context from retrieved sentences
    context_parts = []
    for doc, metadata in zip(docs, metadatas):
        sentence_idx = metadata.get('sentence_idx', 'unknown')
        context_parts.append(f"[Sentence {sentence_idx}] {doc}")

    context = "\n\n".join(context_parts)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Create prompt for RAG
    prompt = f"""You are an expert assistant analyzing a resume/CV document.
Use the following retrieved sentences from the document to answer the user's question.

Retrieved Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Reference specific parts of the resume when relevant
- If the context doesn't contain enough information, say so clearly
- Keep your answer focused and relevant to the resume content

Answer:"""

    # Generate answer
    response = llm.invoke(prompt)
    return response.content


def demonstrate_sentence_retrieval(collection):
    """Demonstrate sentence-level RAG with sample queries"""
    print("\n🤖 Testing Sentence-Level RAG")
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
            answer = rag_query_sentences(query, collection, top_k=5)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")

        print("-" * 50)

def demonstrate_langchain_sentence_rag(vectorstore: Chroma):
    """Demonstrate sentence-level RAG using LangChain RetrievalQA"""
    print("\n🧪 Testing Sentence-Level RAG with LangChain")
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
            answer = rag_query_langchain(vectorstore, query)
            print(f"[green]Answer:[/green]\n{answer}")
        except Exception as e:
            print(f"[red]Error generating answer: {e}[/red]")

        print("-" * 50)

if __name__ == "__main__":
    print("[bold green]📄 Sentence-Level PDF Segmentation with LangChain[/bold green]")
    print("=" * 60)

    pdf_path = "/Users/jyotiprakash/Desktop/python/rag-simulator/study-research/sliding-window/Resume JPM.pdf"

    # Extract sentences from PDF
    print("📝 Segmenting PDF into sentences...")
    sentences = segment_pdf_sentences(pdf_path)
    print(f"✅ Extracted {len(sentences)} sentences")

    # Display sample sentences
    print("\n📋 Sample Sentences:")
    for i, sentence in enumerate(sentences[:5], 1):
        print(f"  {i}. {sentence[:100]}{'...' if len(sentence) > 100 else ''}")

    # Setup traditional ChromaDB
    print("\n🗄️ Setting up ChromaDB...")
    client, collection = setup_chromadb_collection()

    # Store sentences in ChromaDB
    print("💾 Storing sentences in ChromaDB...")
    store_sentences_in_chromadb(sentences, collection, pdf_path)

    # Demonstrate traditional retrieval
    demonstrate_sentence_retrieval(collection)

    # Setup LangChain Chroma
    print("\n🔗 Setting up LangChain Chroma...")
    vectorstore = store_sentences_in_langchain_chroma(sentences)

    # Demonstrate LangChain retrieval
    demonstrate_langchain_sentence_rag(vectorstore)

    print("\n[bold green]🎉 Sentence-level segmentation and RAG demo completed![/bold green]")