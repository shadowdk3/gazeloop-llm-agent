import ollama

"""
Implementing the embedding generation stage of the RAG system.

It converts text chunks into numerical vectors (embedding vectors) so that PostgreSQL 
with pgvector can later search for semantically similar content.
"""

class EmbeddingService:
    """
    Convert text into numerical embedding vectors using an Ollama model.
    """
    
    def __init__(self, model="nomic-embed-text"):
        # nomic-embed-text generates 768-dimensional vectors.
        self.model = model

    def embed(self, text):
        """
        Generate an embedding vector for the input text.

        Args:
            text: A string containing a knowledge chunk or user query.

        Returns:
            A list of floating-point numbers representing the text.
        """
        # Send the text to Ollama's embedding model.
        response = ollama.embeddings(
            model=self.model,
            prompt=text
        )

        # Extract and return the embedding vector from the response.
        return response["embedding"]