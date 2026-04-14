import json
from langsmith import traceable
from src.db import get_connection
from src.embedding import get_embedding
from src.logger import logger

@traceable(name="Postgres PGVector Similarity Search")
async def retrieve_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Requirement 5: Query function performing cosine similarity search via IVF Flat."""
    logger.info(f"Retrieving top {top_k} similar chunks for query.")
    query_embedding = get_embedding(query)
    
    conn = await get_connection()
    results = []
    try:
        # Increase IVFFlat probes to check multiple lists, ensuring results aren't bypassed on small datasets
        await conn.execute("SET ivfflat.probes = 100;")
        
        # Using pgvector's <=> operator for cosine distance. 
        # Cosine similarity is 1 - Cosine Distance.
        sql = """
            SELECT id, content, metadata, 1 - (embedding <=> $1::vector) AS similarity
            FROM document_vectors
            ORDER BY embedding <=> $2::vector
            LIMIT $3;
        """
        
        # asyncpg handles parameter mapping internally
        rows = await conn.fetch(sql, query_embedding, query_embedding, top_k)
        
        for row in rows:
            # Parse raw postgres JSONB string into python dictionary dict
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else dict(row['metadata'])
            
            results.append({
                "id": str(row['id']),
                "content": row['content'],
                "metadata": metadata,
                "similarity": float(row['similarity'])
            })
            
        logger.info(f"Successfully retrieved {len(results)} chunks.")
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise
    finally:
        await conn.close()
        
    return results
