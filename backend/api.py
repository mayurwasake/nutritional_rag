from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from src.db import init_db, batch_insert_embeddings
from src.ingest import parse_pdf_parallel, chunk_documents
from src.embedding import get_embeddings
from src.retrieve import retrieve_similar_chunks
from src.chat import generate_rag_response

app = FastAPI(title="Nutritional RAG API", description="Production-Grade FastAPI Backend for PDF RAG System")

# Production CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: In strict production (public internet), replace "*" with your frontend domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ends up chunking the document and placing it into PGVector."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Reset DB before ingesting the new file limits hallucinations from previous files
        init_db(reset=True)
        
        # Run local ingestion pipeline
        pages_data = parse_pdf_parallel(tmp_path)
        chunks_data = chunk_documents(pages_data)
        
        contents = [c["content"] for c in chunks_data]
        metadatas = [c["metadata"] for c in chunks_data]
        
        # High speed local vector generation
        embeddings = get_embeddings(contents)
        batch_insert_embeddings(contents, embeddings, metadatas)
        
        # Cleanup temp file
        os.remove(tmp_path)
        
        return {
            "message": f"Document {file.filename} ingested successfully", 
            "chunks_processed": len(chunks_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Executes the standard RAG logic."""
    try:
        # 1. Retrieve the vector chunks (Local embeddings)
        results = retrieve_similar_chunks(request.query, top_k=5)
        
        if not results:
            return ChatResponse(
                answer="No relevant information found in the document to index against.", 
                sources=[]
            )
        
        # 2. RAG Generation (Gemini LLM)
        answer = generate_rag_response(request.query, results)
        
        return ChatResponse(
            answer=answer,
            sources=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
