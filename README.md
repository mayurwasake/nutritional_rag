# High-Speed RAG Ingestion and Retrieval Pipeline

This repository implements a production-level RAG codebase optimized for large volumes of text (like a 1,200-page textbook).

## Project Achievements & Features
- **Parallel Ingestion Layer**: Uses Python's `ProcessPoolExecutor` alongside PyMuPDF (`fitz`) for extremely fast parsing of massive PDFs.
- **Recursive Chunking**: Integrates LangChain's `RecursiveCharacterTextSplitter` (1000 chunk size / 100 overlap).
- **Vector Database**: PostgreSQL combined with the `pgvector` extension.
- **Latency Optimization**: Uses `ivfflat` indexing configured with `vector_cosine_ops` and `lists=100` for highly optimized query retrieval speed.

## Prerequisites
- **Python**: 3.10+
- **PostgreSQL**: Running instance with the `pgvector` extension installed.
- **Gemini API Key**: Used for generating embeddings.

## Setup Instructions

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Environment Configuration**
Copy the example environment variables file and configure it:
```bash
cp .env.example .env
```
Ensure you have created the correct database and the postgres credentials match. Add your Gemini API Key to `.env`.

3. **Run the Streamlit App (Frontend + Backend interface)**
```bash
streamlit run app.py
```
This will open a web interface in your browser where you can manually upload PDFs for ingestion and type queries into a text box to get relevant semantic chunks.

### Alternative Command Line Usage
**Ingest Textbook**
```bash
python main.py --mode ingest --pdf /path/to/your/1200_page_textbook.pdf
```

**Query & Retrieve**
```bash
python main.py --mode query --query "What are the macronutrients discussed in chapter 1?"
```
