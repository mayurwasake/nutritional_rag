# 📚 High-Speed RAG Ingestion and Retrieval Pipeline (Production Version)

This repository implements a production-grade, microservices-based Retrieval-Augmented Generation (RAG) codebase optimized for large volumes of text (like a 1,200-page textbook).

## 🏗️ Architecture Overview
The project has been refactored into a scalable, Dockerized microservices architecture:
- **Frontend (Streamlit)**: A clean, isolated UI for uploading documents and chatting.
- **Backend (FastAPI)**: Handles heavy lifting (PDF parsing, chunking, embedding, and LLM synthesis) via REST endpoints, preventing the UI from freezing.
- **Database (PostgreSQL + pgvector)**: Stores and queries dense vector embeddings at lightning speed.

## ✨ Key Features
- **High-Speed Local Embeddings**: Uses CPU-optimized `sentence-transformers` (`all-MiniLM-L6-v2`) locally to bypass cloud rate limits and reduce ingestion costs to $0.
- **Generative AI Chat**: Integrates `google-genai` (Gemini 2.5 Flash) strictly for conversational RAG synthesis over the retrieved text chunks.
- **Parallel Ingestion**: Uses Python's `ThreadPoolExecutor` alongside PyMuPDF (`fitz`) for extremely fast parsing of massive PDFs.
- **Database Auto-Reset**: Truncates old embeddings automatically upon new uploads to prevent "database contamination" (cross-document hallucinations).
- **One-Click Deployment**: Managed entirely via `docker-compose`.

## 🚀 Quick Start (Docker)

### 1. Configure Environment
Create your `.env` file and add your Gemini API Key:
```bash
cp .env.example .env
```
Open `.env` and paste your actual key: `GEMINI_API_KEY=your_key_here`

### 2. Launch the Application
Start the entire stack (Database, Backend API, and Frontend) with a single command:
```bash
docker-compose up --build
```

### 3. Access the Interfaces
Once the terminal logs show "Application startup complete", you can visit:
- **Frontend UI (Streamlit)**: [http://localhost:8501](http://localhost:8501)
- **Backend API Interactive Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📂 Project Structure
```text
nutritional_rag/
├── docker-compose.yml 
├── .env & .env.example
├── frontend/                     # Streamlit UI service
│   ├── app.py 
│   ├── requirements.txt          
│   └── Dockerfile                
└── backend/                      # FastAPI service & RAG Engine
    ├── api.py  
    ├── src/                      # Database, chunking, and embedding logic
    ├── requirements.txt          
    └── Dockerfile                
```
