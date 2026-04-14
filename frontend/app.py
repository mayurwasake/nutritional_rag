import streamlit as st
import time
import requests
import os

# Production grade fallback for environment variables
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Textbook RAG System", page_icon="📚", layout="wide")

st.title("📚 High-Speed Textbook RAG System")
st.markdown("Upload a large PDF textbook and start asking questions!")

# Sidebar for Upload and Processing
with st.sidebar:
    st.header("1. Upload & Ingest")
    uploaded_file = st.file_uploader("Upload your PDF textbook", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Uploading and Processing Document..."):
                try:
                    start_time = time.time()
                    
                    # Send File to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"{data['message']} ({data['chunks_processed']} chunks)")
                        st.text(f"Total time taken: {time.time() - start_time:.2f}s")
                    else:
                        st.error(f"Error processing document: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.error("Please upload a PDF file first.")

# Main Area for Chat / Query
st.header("Ask Questions")

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
                # Issue HTTP POST request to FastAPI backend
                response = requests.post(f"{API_URL}/chat", json={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    results = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if results:
                        # 3. Add sources explicitly into an expander to maintain trust/credibility
                        with st.expander("View Source Chunks"):
                            for i, res in enumerate(results, 1):
                                # FastAPI json parsing dicts safely
                                metadata = res.get('metadata', {})
                                page_num = metadata.get('page_num', 'N/A')
                                similarity = res.get('similarity', 0.0)
                                content = res.get('content', '')
                                
                                st.write(f"**Source {i}** (Page {page_num} | Similarity: {similarity:.4f})")
                                st.write(content)
                                st.divider()
                    else:
                        st.warning("No relevant information found in the document to index against.")
                        
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error from backend: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
           