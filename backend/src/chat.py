from google import genai
from langsmith import traceable
from src.config import GEMINI_API_KEY

@traceable(name="LLM RAG Generation (Gemini)")
def generate_rag_response(query: str, contexts: list[dict]) -> str:
    """
    Takes the user query and the retrieved database chunks
    and uses Gemini to formulate a conversational, intelligent RAG response.
    """
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file to enable chatting."
    
    # Format the retrieved chunks into a clean context string
    formatted_context = ""
    for idx, c in enumerate(contexts, 1):
        formatted_context += f"--- Source {idx} (Page {c['metadata'].get('page_num', 'N/A')}) ---\n"
        formatted_context += f"{c['content']}\n\n"
        
    prompt = f"""
    You are an intelligent, helpful Textbook Assistant. 
    Use ONLY the provided Context below to answer the User's question. 
    Do not use outside knowledge or hallucinate information that is not present in the Context.
    If the answer cannot be found in the provided Context, politely say that you don't know based on the provided document.
    
    Context:
    {formatted_context}
    
    User Question: {query}
    
    Answer:
    """
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error generating response from LLM: {str(e)}"