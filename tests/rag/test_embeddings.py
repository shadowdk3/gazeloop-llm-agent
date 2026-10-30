from unittest.mock import patch

from src.rag.embeddings import EmbeddingService

"""
verify embedding service can successfully convert text into a numerical embedding vector
"""

# temporarily replaces the real Ollama embedding function with a mock.
@patch("src.rag.embeddings.ollama.embeddings")
def test_embedding(mock_embeddings):
    """Verify that the embedding service generates a valid vector."""

    expected_vector = [0.1] * 768

    # Simulate the response returned by Ollama
    mock_embeddings.return_value = {
        "embedding": expected_vector
    }
    
    embedder = EmbeddingService()

    text = (
        "A confirmed fall should be recorded "
        "in the incident management system."
    )

    vector = embedder.embed(text)

    # Check that the embedding is not empty
    assert vector is not None
    assert len(vector) > 0

    # Check expected embedding dimensions
    assert len(vector) == 768

    # Check that all values are numerical
    assert all(isinstance(value, (int, float)) for value in vector)

    # Check that the vector contains valid numerical values
    assert all(
        isinstance(value, (int, float)) and
        value == value and
        abs(value) != float("inf")
        for value in vector
    )

    mock_embeddings.assert_called_once_with(
        model="nomic-embed-text",
        prompt=text
    )
    
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])
    
@patch("src.rag.embeddings.ollama.embeddings")
def test_embedding_consistency(mock_embeddings):
    """
    check whether the same text produces consistent embeddings.
    """
    
    expected_vector = [0.2] * 768

    mock_embeddings.return_value = {
        "embedding": expected_vector
    }
    
    embedder = EmbeddingService()

    text = "A person has fallen on the floor."

    vector1 = embedder.embed(text)
    vector2 = embedder.embed(text)

    assert len(vector1) == len(vector2)

    for value1, value2 in zip(vector1, vector2):
        assert abs(value1 - value2) < 1e-6