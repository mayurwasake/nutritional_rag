import os
import time
from src.db import init_db, batch_insert_embeddings
from src.ingest import parse_pdf_parallel, chunk_documents
from src.embedding import get_embeddings
from src.retrieve import retrieve_similar_chunks

def run_ingestion(pdf_path: str):
    """Orchestrates the entire ingestion and indexing pipeline."""
    print("=== Step 1: Initializing Database Schema & IVF Flat Index ===")
    init_db()
    
    print(f"\n=== Step 2: Parallel PDF Parsing ({pdf_path}) ===")
    start_time = time.time()
    pages_data = parse_pdf_parallel(pdf_path)
    print(f"Parsing took {time.time() - start_time:.2f} seconds.")
    
    print("\n=== Step 3: Recursive Chunking ===")
    chunks_data = chunk_documents(pages_data)
    
    print("\n=== Step 4: Generating Embeddings in Batches ===")
    start_time = time.time()
    contents = [c["content"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]
    
    # Generate embeddings (Local HuggingFace model via sentence-transformers)
    embeddings = get_embeddings(contents)
    print(f"Embedding {len(contents)} chunks took {time.time() - start_time:.2f} seconds.")
    
    print("\n=== Step 5: Batch Inserting to PGVector ===")
    start_time = time.time()
    batch_insert_embeddings(contents, embeddings, metadatas)
    print(f"Insertion took {time.time() - start_time:.2f} seconds.")

def run_retrieval(query: str):
    print(f"\n=== Retrieving Context for: '{query}' ===")
    start_time = time.time()
    results = retrieve_similar_chunks(query, top_k=5)
    print(f"Retrieval took {time.time() - start_time:.4f} seconds.")
    
    for i, res in enumerate(results, 1):
        score = res['similarity']
        page = res['metadata'].get('page_num', 'N/A')
        print(f"\n[Result {i}] Score: {score:.4f} | Page: {page}")
        print("-" * 50)
        print(res['content'][:300] + "...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Speed RAG Pipeline")
    parser.add_argument("--mode", choices=["ingest", "query"], required=True)
    parser.add_argument("--pdf", type=str, help="Path to PDF (Required for ingest)")
    parser.add_argument("--query", type=str, help="Query string (Required for search)")
    
    args = parser.parse_args()
    
    if args.mode == "ingest":
        if not args.pdf:
            print("Error: --pdf path is required for ingestion.")
        else:
            run_ingestion(args.pdf)
    elif args.mode == "query":
        if not args.query:
            print("Error: --query string is required for retrieval.")
        else:
            run_retrieval(args.query)
