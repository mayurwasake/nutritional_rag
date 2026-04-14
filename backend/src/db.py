import asyncpg
import json
from pgvector.asyncpg import register_vector
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_DIM
from src.logger import logger

async def get_connection(setup=False):
    """Establish an async connection to PostgreSQL and register pgvector."""
    try:
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        if setup:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
        await register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

async def init_db(reset: bool = False):
    """Requirement 3 & 4: Create schema and IVF Flat index."""
    logger.info(f"Initializing database schema. Reset={reset}")
    conn = await get_connection(setup=True)
    try:
        # R3: Create the document_vectors table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS document_vectors (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        
        if reset:
            logger.warning("Truncating document_vectors table...")
            await conn.execute("TRUNCATE TABLE document_vectors RESTART IDENTITY;")
            
    finally:
        await conn.close()

async def batch_insert_embeddings(chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """Requirement 3: Batch-insert embeddings using high-speed async executemany."""
    logger.info(f"Batch inserting {len(chunks)} text chunks into Vector DB.")
    conn = await get_connection()
    try:
        query = """
            INSERT INTO document_vectors (content, metadata, embedding)
            VALUES ($1, $2, $3)
        """
        
        # Prepare data tuples formatting metadata dicts to JSON strings for Postgres
        data = [(c, json.dumps(m), e) for c, m, e in zip(chunks, metadatas, embeddings)]
        
        # Execute rapid batch insert via asyncpg
        await conn.executemany(query, data)
        
        # R4: Create IVFFlat index AFTER data is inserted
        total_rows = await conn.fetchval("SELECT count(*) FROM document_vectors;")
        
        # Standard pgvector rule: Lists shouldn't exceed rows. Set 100 for large DB, clamped for small test docs.
        lists = 100 if total_rows >= 100 else max(1, total_rows // 2)
        
        logger.info(f"Rebuilding IVFFlat Index with lists={lists} for {total_rows} total database vectors.")
        await conn.execute("DROP INDEX IF EXISTS document_vectors_embedding_idx;")
        await conn.execute(f"""
            CREATE INDEX document_vectors_embedding_idx
            ON document_vectors USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)
    except Exception as e:
        logger.error(f"Error during batch insertion: {e}")
        raise
    finally:
        await conn.close()
