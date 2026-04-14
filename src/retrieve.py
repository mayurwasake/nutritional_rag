from src.db import get_connection
from src.embedding import get_embedding

def retrieve_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Requirement 5: Query function performing cosine similarity search via IVF Flat."""
    query_embedding = get_embedding(query)
    
    conn = get_connection()
    results = []
    try:
        with conn.cursor() as cur:
            # Using pgvector's <=> operator for cosine distance. 
            # Cosine similarity is 1 - Cosine Distance.
            
            # Increase IVFFlat probes to check multiple lists, ensuring results aren't bypassed on small datasets
            cur.execute("SET ivfflat.probes = 100;")
            
            sql = """
                SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
                FROM document_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            
            # psycopg2 will adapt the Python list to Postgres array automatically
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            
            rows = cur.fetchall()
            for row in rows:
                results.append({
                    "id": str(row[0]),
                    "content": row[1],
                    "metadata": row[2],
                    "similarity": float(row[3])
                })
    finally:
        conn.close()
        
    return results
