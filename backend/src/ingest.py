import fitz  # PyMuPDF
import concurrent.futures
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_page(pdf_path: str, page_num: int) -> dict:
    """Extracts text from a single PDF page."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        text = page.get_text("text")
        doc.close()
        return {"page_num": page_num, "text": text}
    except Exception as e:
        print(f"Error reading page {page_num}: {e}")
        return {"page_num": page_num, "text": ""}

def parse_pdf_parallel(pdf_path: str, max_workers: int = None) -> list[dict]:
    """Requirement 1: Parallel Ingestion Layer using PyMuPDF and ThreadPoolExecutor for Streamlit."""
    if max_workers is None:
        max_workers = os.cpu_count() or 4
        
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    print(f"Starting parallel extraction of {num_pages} pages using {max_workers} threads...")
    pages_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks mapped to PDF pages
        futures = {executor.submit(extract_text_from_page, pdf_path, i): i for i in range(num_pages)}

        for future in concurrent.futures.as_completed(futures):
            pages_data.append(future.result())

    # Sort to maintain ascending page order naturally
    pages_data.sort(key=lambda x: x["page_num"])
    return pages_data

def chunk_documents(pages_data: list[dict]) -> list[dict]:
    """Requirement 2: LangChain's RecursiveCharacterTextSplitter."""
    print(f"Splitting text with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for page in pages_data:
        text = page.get("text", "").strip()
        if not text:
            continue
            
        page_chunks = splitter.split_text(text)
        
        for chunk in page_chunks:
            chunks.append({
                "content": chunk,
                "metadata": {"page_num": page["page_num"]}
            })
            
    print(f"Generated {len(chunks)} chunks.")
    return chunks
