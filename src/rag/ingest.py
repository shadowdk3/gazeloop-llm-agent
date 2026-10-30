from src.rag.chunking import load_document, chunk_by_section
from src.rag.embeddings import EmbeddingService

from pathlib import Path
import psycopg2
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DOCUMENT = PROJECT_DIR / "knowledge" / "fall_response.md"

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.

    Connection settings are read from environment variables.
    Default values are used if the variables are not configured.
    """
    
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "gazeloop"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("PGPASSWORD", "123456"),
    )

def ingest_document():
    """
    Load a knowledge document, generate embeddings for its sections,
    and store the chunks and embeddings in PostgreSQL.
    """
    
    # Establish a database connection.
    conn = get_db_connection()

    try:
        cursor = conn.cursor()
        
        # Use the document filename to identify existing records.
        document_name = DOCUMENT.name

        # Check whether this document has already been ingested.
        # EXISTS returns True if at least one matching record is found.
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM knowledge_chunks
                WHERE document_name = %s
            )
            """,
            (document_name,)
        )
        
        document_exists = cursor.fetchone()[0]
        
        # Skip ingestion if the document already exists.
        # This prevents generating duplicate chunks and embeddings.
        if document_exists:
            print(f"{document_name} already exists. Skipping ingestion.")
            return
        
        # Read the Markdown document into a string.
        text = load_document(DOCUMENT)

        # Split the document into sections using level-2 headings.
        # Each chunk contains a section title and its corresponding content.
        chunks = chunk_by_section(text)

        # Initialize the embedding service.
        # The default model is nomic-embed-text.
        embedder = EmbeddingService()

        # Process each section individually.
        for chunk in chunks:
            # Extract the section content.
            content = chunk["content"]

            # Combine the section title and content before embedding.
            # Including the title provides additional context to the model.
            embedding_text = (
                f"Section: {chunk['section']}\n\n"
                f"{content}"
            )

            # Convert the text into a numerical embedding vector.
            embedding = embedder.embed(embedding_text)

            # Insert the document information and embedding into PostgreSQL.
            # The %s placeholders are parameterized to safely pass values.
            cursor.execute(
                """
                INSERT INTO knowledge_chunks
                (
                    document_name,
                    section,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    document_name,
                    chunk["section"],
                    content,
                    embedding
                )
            )

        # Commit the transaction to permanently save all inserted records.
        conn.commit()
        print(f"Successfully ingested {len(chunks)} chunks.")
        
    except Exception:
        # Roll back all inserts if an error occurs.
        # This prevents a partially ingested document.
        conn.rollback()
        raise
    
    finally:
        # Close the database connection, whether ingestion succeeds or fails.
        conn.close()

if __name__ == "__main__":
    ingest_document()