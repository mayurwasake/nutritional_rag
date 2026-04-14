import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_DIM

def get_connection(setup=False):
    """Establish a connection to PostgreSQL and register pgvector."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    if setup:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
    register_vector(conn)
    return conn

def init_db(reset: bool = False):
    """Requirement 3 & 4: Create schema and IVF Flat index."""
    conn = get_connection(setup=True)
    try:
        with conn.cursor() as cur:

            # R3: Create the document_vectors table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR({EMBEDDING_DIM})
                );
            """)
            
            if reset:
                cur.execute("TRUNCATE TABLE document_vectors RESTART IDENTITY;")
                
        conn.commit()
    finally:
        conn.close()

def batch_insert_embeddings(chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """Requirement 3: Batch-insert embeddings into PostgreSQL."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO document_vectors (content, metadata, embedding)
                VALUES %s
            """
            
            # Prepare data tuples in the correct format
            data = [(c, psycopg2.extras.Json(m), e) for c, m, e in zip(chunks, metadatas, embeddings)]
            
            # Execute batch insert using psycopg2's execute_values
            execute_values(cur, query, data, page_size=100)
            
            # R4: Create IVFFlat index AFTER data is inserted
            cur.execute("SELECT count(*) FROM document_vectors;")
            total_rows = cur.fetchone()[0]
            
            # Standard pgvector rule: Lists shouldn't exceed rows. Set 100 for large DB, clamped for small test docs.
            lists = 100 if total_rows >= 100 else max(1, total_rows // 2)
            
            cur.execute("DROP INDEX IF EXISTS document_vectors_embedding_idx;")
            cur.execute(f"""
                CREATE INDEX document_vectors_embedding_idx
                ON document_vectors USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
            
        conn.commit()
    finally:
        conn.close()
