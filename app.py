import streamlit as st
import tempfile
import os
import time
from src.db import init_db, batch_insert_embeddings
from src.ingest import parse_pdf_parallel, chunk_documents
from src.embedding import get_embeddings
from src.retrieve import retrieve_similar_chunks
from src.chat import generate_rag_response

st.set_page_config(page_title="Textbook RAG System", page_icon="📚", layout="wide")

st.title("📚 High-Speed Textbook RAG System")
st.markdown("Upload a large PDF textbook and start asking questions!")

# Sidebar for Upload and Processing
with st.sidebar:
    st.header("1. Upload & Ingest")
    uploaded_file = st.file_uploader("Upload your PDF textbook", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
                
            st.info("Initializing Database...")
            init_db()
            
            with st.spinner("Extracting text in parallel..."):
                start_time = time.time()
                pages_data = parse_pdf_parallel(tmp_path)
                st.text(f"Extraction took {time.time() - start_time:.2f}s")
            
            with st.spinner("Chunking document..."):
                chunks_data = chunk_documents(pages_data)
                st.text(f"Generated {len(chunks_data)} chunks")
                
            contents = [c["content"] for c in chunks_data]
            metadatas = [c["metadata"] for c in chunks_data]
            
            with st.spinner("Generating embeddings locally..."):
                start_time = time.time()
                # Local sentence-transformers generates embeddings extremely fast via internal batching
                embeddings = get_embeddings(contents)
                
                st.text(f"Embedding took {time.time() - start_time:.2f}s")
                
            with st.spinner("Inserting into PGVector..."):
                start_time = time.time()
                batch_insert_embeddings(contents, embeddings, metadatas)
                st.text(f"Database insertion took {time.time() - start_time:.2f}s")
                
            st.success("Document ingestion complete!")
            # Clean up temp file
            os.remove(tmp_path)
        else:
            st.error("Please upload a PDF file first.")

# Main Area for Chat / Query
st.header("2. Ask Questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve the vector chunks (Local embeddings)
                results = retrieve_similar_chunks(prompt, top_k=5)
                
                if not results:
                    st.warning("No relevant information found in the document to index against.")
                else:
                    # 2. RAG Generation (Gemini LLM)
                    answer = generate_rag_response(prompt, results)
                    st.markdown(answer)
                    
                    # 3. Add sources explicitly into an expander to maintain trust/credibility
                    with st.expander("View Source Chunks"):
                        for i, res in enumerate(results, 1):
                            page_num = res['metadata'].get('page_num', 'N/A')
                            st.write(f"**Source {i}** (Page {page_num} | Similarity: {res['similarity']:.4f})")
                            st.write(res['content'])
                            st.divider()
                            
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                st.error(f"Error during retrieval/generation: {e}")
           