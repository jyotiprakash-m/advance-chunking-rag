import os
import pdfplumber
import nltk
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List
from tqdm import tqdm
import concurrent.futures

load_dotenv()
nltk.download("punkt")


def process_page_text(page_text: str) -> List[str]:
    """Process page text to extract structural blocks."""
    if page_text:
        return [block.strip() for block in page_text.split("\n\n") if block.strip()]
    return []


def extract_and_store_blocks_streaming(
    pdf_path: str,
    collection_name: str = "bigfile_structural_blocks",
    max_workers: int = 4,
    batch_size: int = 50,
):
    """
    Extract structural blocks from a PDF with streaming + parallel processing,
    and store them in ChromaDB as they are created.
    Shows progress with tqdm.
    """
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding)

    with pdfplumber.open(pdf_path) as pdf:
        pages = list(pdf.pages)
        total_pages = len(pages)
        print(f"Total pages to process: {total_pages}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(process_page_text, page.extract_text()): i
                for i, page in enumerate(pages)
            }

            buffer = []
            block_index = 0

            # Progress bar for page processing
            for future in tqdm(
                concurrent.futures.as_completed(future_to_page),
                total=total_pages,
                desc="Processing pages",
                unit="page",
            ):
                page_blocks = future.result()

                # Progress bar for blocks inside each page
                for block in tqdm(
                    page_blocks,
                    leave=False,
                    desc="Storing blocks",
                    unit="block",
                ):
                    buffer.append(Document(page_content=block, metadata={"block_index": block_index}))
                    block_index += 1

                    if len(buffer) >= batch_size:
                        vectorstore.add_documents(buffer)
                        buffer.clear()

            # Flush remaining docs
            if buffer:
                vectorstore.add_documents(buffer)

    print(f"✅ Stored {block_index} blocks in ChromaDB.")
    return vectorstore


if __name__ == "__main__":
    pdf_path = "/Users/jyotiprakash/Desktop/python/rag-simulator/big_file/bigfile.pdf"
    extract_and_store_blocks_streaming(pdf_path, max_workers=4, batch_size=50)
