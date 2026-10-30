from unittest.mock import MagicMock, patch

from src.rag import ingest

def test_ingest_document_success():
    """
    Test that document chunks are embedded and inserted into the database.
    """

    # Mock the PostgreSQL connection and cursor.
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # Simulate that the document does not already exist.
    mock_cursor.fetchone.return_value = (False,)

    # Provide two sample chunks.
    sample_chunks = [
        {
            "section": "1. Incident Verification",
            "content": "Verify whether a person has fallen."
        },
        {
            "section": "2. Immediate Response",
            "content": "Check whether the person needs assistance."
        }
    ]

    # Mock the embedding service.
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 768

    with patch.object(ingest, "get_db_connection", return_value=mock_conn), \
         patch.object(ingest, "load_document", return_value="sample document"), \
         patch.object(ingest, "chunk_by_section", return_value=sample_chunks), \
         patch.object(ingest, "EmbeddingService", return_value=mock_embedder):

        ingest.ingest_document()

    # Check that two embeddings were generated.
    assert mock_embedder.embed.call_count == 2

    # Check that two SQL INSERT statements were executed.
    insert_calls = [
        call for call in mock_cursor.execute.call_args_list
        if "INSERT INTO knowledge_chunks" in call.args[0]
    ]
    assert len(insert_calls) == 2

    # Check that the database transaction was committed.
    mock_conn.commit.assert_called_once()

    # Check that the database connection was closed.
    mock_conn.close.assert_called_once()


def test_ingest_document_skips_existing_document():
    """
    Test that ingestion is skipped when the document already exists.
    """

    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value

    # Simulate an existing document.
    mock_cursor.fetchone.return_value = (True,)

    with patch.object(ingest, "get_db_connection", return_value=mock_conn), \
         patch.object(ingest, "EmbeddingService") as mock_embedder:

        ingest.ingest_document()

    # No embedding should be generated.
    mock_embedder.assert_not_called()

    # No new records should be inserted.
    insert_calls = [
        call for call in mock_cursor.execute.call_args_list
        if "INSERT INTO knowledge_chunks" in call.args[0]
    ]
    assert len(insert_calls) == 0

    # No database changes should be committed.
    mock_conn.commit.assert_not_called()

    mock_conn.close.assert_called_once()