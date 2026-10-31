import os

import psycopg2

from src.rag.embeddings import EmbeddingService
from src.rag.models import RetrievedChunk

class RAGRetriever:
    """
    Retrieve the most relevant safety guidelines from the database.

    When an accident risk is detected:
    1. Convert the incident description into an embedding.
    2. Search for similar embeddings in PostgreSQL.
    3. Return the most relevant safety guideline chunks.
    """
    
    def __init__(self, embedding_model="nomic-embed-text"):
        # Initialize the embedding service.
        # The same embedding model should be used during ingestion
        # and retrieval so that their vectors are compatible.
        self.embedder = EmbeddingService(model=embedding_model)

    def get_db_connection(self):
        """Create and return a PostgreSQL database connection."""
        
        # Read database settings from environment variables.
        # Use the default values if the variables are not configured.
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "gazeloop"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("PGPASSWORD", "123456"),
        )

    def retrieve(self, query: str, top_k: int = 3):
        """
        Find the top_k most relevant knowledge chunks for a query.

        Args:
            query: Incident description or safety-related question.
            top_k: Maximum number of chunks to retrieve (default: 3).

        Returns:
            A list of RetrievedChunk objects.
        """

        # Step 1: Convert the incident description into an embedding vector.
        # This represents the semantic meaning of the query numerically.
        query_embedding = self.embedder.embed(query)

        # Step 2: Connect to PostgreSQL.
        conn = self.get_db_connection()
        try:
            # Create a cursor to execute SQL statements.
            cursor = conn.cursor()

            # Step 3: Search the knowledge_chunks table.
            cursor.execute(
                """
                SELECT
                    document_name,
                    section,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    top_k,
                ),
            )

            # Step 4: Fetch all rows returned by the SQL query.
            results = cursor.fetchall()

        finally:
            # Step 5: Close the database connection even if an error occurs.
            conn.close()
            
        # Step 6: Convert database rows into RetrievedChunk objects.
        # This gives the rest of the RAG pipeline a consistent data structure.
        return [
            RetrievedChunk(
                document_name=row[0],
                section=row[1],
                content=row[2],
                similarity=float(row[3]),
            )
            for row in results
        ]