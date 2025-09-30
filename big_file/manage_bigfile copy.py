import os
import pdfplumber
import nltk
from dotenv import load_dotenv
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain.docstore.document import Document
from typing import List, Tuple
from tqdm import tqdm
import concurrent.futures
import time

load_dotenv()
 
# Download NLTK data only once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)


def process_page_text(page_data: Tuple[int, str]) -> List[Tuple[int, str]]:
    """
    Split page text into ~50-word sentence-aware blocks.
    Returns list of (page_num, block_text) tuples.
    """
    page_num, page_text = page_data
    
    if not page_text or not page_text.strip():
        return []
    
    sentences = nltk.sent_tokenize(page_text)
    blocks = []
    current_block = ""
    
    for sentence in sentences:
        current_block += sentence + " "
        if len(current_block.split()) >= 50:
            blocks.append((page_num, current_block.strip()))
            current_block = ""
    
    if current_block.strip():
        blocks.append((page_num, current_block.strip()))
    
    return blocks


def extract_blocks_streaming(
    pdf_path: str,
    max_workers: int = 4,
    batch_size: int = 50,
    page_batch_size: int = 100,
):
    """Process all pages in batches, create blocks, but do NOT store or embed."""
    start_time = time.time()
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # embedding = OpenAIEmbeddings()
    # vectorstore = Chroma(
    #     collection_name=collection_name,
    #     embedding_function=embedding,
    #     persist_directory="./chroma_db_bigfile"
    # )

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"📄 Total pages to process: {total_pages}")

        buffer = []
        block_index = 0
        batch_count = 0
        total_blocks_processed = 0

        # Process pages in batches with progress bar
        for page_start in tqdm(range(0, total_pages, page_batch_size), 
                               desc="Processing page batches", 
                               unit="batch"):
            page_end = min(page_start + page_batch_size, total_pages)
            
            # Extract text from pages (do this in main process to avoid pickling issues)
            page_data = [
                (page_start + i, page.extract_text() or "")
                for i, page in enumerate(pdf.pages[page_start:page_end])
            ]

            # Process pages in parallel using ThreadPoolExecutor instead of ProcessPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {
                    executor.submit(process_page_text, pd): pd[0]
                    for pd in page_data
                }

                # Collect results in order
                page_results = {}
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        page_blocks = future.result()
                        page_results[page_num] = page_blocks
                    except Exception as e:
                        print(f"⚠️ Error processing page {page_num + 1}: {e}")
                        page_results[page_num] = []

                # Add blocks in page order
                for page_num in sorted(page_results.keys()):
                    for page_num, block_text in page_results[page_num]:
                        buffer.append(Document(
                            page_content=block_text,
                            metadata={
                                "block_index": block_index,
                                "page_number": page_num + 1,
                                "source": os.path.basename(pdf_path)
                            }
                        ))
                        block_index += 1
                        total_blocks_processed += 1

                        # Just count batches, do not store
                        if len(buffer) >= batch_size:
                            batch_count += 1
                            # vectorstore.add_documents(buffer)
                            buffer.clear()

        # Flush remaining blocks
        if buffer:
            batch_count += 1
            # vectorstore.add_documents(buffer)
            print(f"💾 Processed final batch {batch_count} ({len(buffer)} blocks)")

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total blocks created: {total_blocks_processed}")
    print(f"📦 Total batches processed: {batch_count}")
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")
    print(f"⚡ Average speed: {total_blocks_processed / total_time:.2f} blocks/second")
    
    return buffer


if __name__ == "__main__":
    pdf_path = "/Users/jyotiprakash/Desktop/python/rag-simulator/big_file/bigfile.pdf"  # Replace with your actual PDF path
    
    # Only process and list blocks, do NOT store or create embeddings
    blocks = extract_blocks_streaming(
        pdf_path, 
        max_workers=4, 
        batch_size=50, 
        page_batch_size=100
    )
    print(f"\nSample processed blocks (first 3):")
    for doc in blocks[:3]:
        print(f"Page {doc.metadata['page_number']} | Block {doc.metadata['block_index']}: {doc.page_content[:100]}...")