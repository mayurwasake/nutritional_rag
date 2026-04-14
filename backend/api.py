from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import tempfile
import os

from src.db import init_db, batch_insert_embeddings
from src.ingest import parse_pdf_parallel, chunk_documents
from src.embedding import get_embeddings
from src.retrieve import retrieve_similar_chunks
from src.chat import generate_rag_response
from src.logger import logger

app = FastAPI(title="Nutritional RAG API", description="Production-Grade FastAPI Backend for PDF RAG System")

# Production CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: In strict production (public internet), replace "*" with your frontend domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limit Max Upload Size roughly to 50MB for security
MAX_FILE_SIZE = 50 * 1024 * 1024

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
    logger.info(f"Incoming file upload: {file.filename}")
    
    if not file.filename.lower().endswith(".pdf"):
        logger.error(f"Rejected non-PDF file: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        content = await file.read()
        
        # Enforce file size limits
        if len(content) > MAX_FILE_SIZE:
            logger.error(f"File {file.filename} exceeded maximum size limits ({len(content)} bytes).")
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
            
        # Save uploaded file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"File saved securely to temp path: {tmp_path}")

        # Reset DB before ingesting the new file limits hallucinations from previous files
        await init_db(reset=True)
        
        # Run highly-intensive synchronous machine learning code in standard threadpools 
        # to guarantee the FastAPI server does not block concurrent asynchronous users
        pages_data = await run_in_threadpool(parse_pdf_parallel, tmp_path)
        chunks_data = await run_in_threadpool(chunk_documents, pages_data)
        
        contents = [c["content"] for c in chunks_data]
        metadatas = [c["metadata"] for c in chunks_data]
        
        logger.info(f"Generating vectors for {len(chunks_data)} chunks...")
        embeddings = await run_in_threadpool(get_embeddings, contents)
        
        logger.info("Vectors generated safely, routing to database...")
        await batch_insert_embeddings(contents, embeddings, metadatas)
        
        # Cleanup temp file
        os.remove(tmp_path)
        logger.info(f"Ingestion successful for {file.filename}")
        
        return {
            "message": f"Document {file.filename} ingested successfully", 
            "chunks_processed": len(chunks_data)
        }
    except Exception as e:
        logger.exception("Upload processing failed entirely!")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Executes the standard RAG logic."""
    logger.info(f"Processing query: '{request.query}'")
    try:
        # 1. Retrieve the vector chunks via asyncpg
        results = await retrieve_similar_chunks(request.query, top_k=5)
        
        if not results:
            logger.warning("No context chunks discovered matching the query sentiment.")
            return ChatResponse(
                answer="No relevant information found in the document to index against.", 
                sources=[]
            )
        
        # 2. Synchronous RAG Generation wrapped dynamically 
        logger.info(f"Retrieved {len(results)} chunks. Sending prompt to LLM...")
        answer = await run_in_threadpool(generate_rag_response, request.query, results)
        
        logger.info("Successfully returned LLM response context.")
        return ChatResponse(
            answer=answer,
            sources=results
        )
    except Exception as e:
        logger.exception(f"Chat generation failure triggered: {e}")
        raise HTTPException(status_code=500, detail=str(e))
