import pytest

from src.rag.chunking import chunk_by_section

# Test 1: Split a Markdown document into multiple sections
def test_chunk_by_section():
    text = """
    # Fall Response Guide

    Introduction text.

    ## Assess the Situation

    Check whether the person is conscious.

    ## Call for Help

    Contact emergency services.
    """

    chunks = chunk_by_section(text)

    assert len(chunks) == 2

    assert chunks[0]["section"] == "Assess the Situation"
    assert chunks[0]["content"] == "Check whether the person is conscious."

    assert chunks[1]["section"] == "Call for Help"
    assert chunks[1]["content"] == "Contact emergency services."

# Test 2: Empty document should return an empty list
def test_chunk_empty_document():
    text = ""

    chunks = chunk_by_section(text)

    assert chunks == []

# Test 3: Ignore sections without content
def test_chunk_section_without_content():
    text = """
    ## Assess the Situation

    ## Call for Help

    Contact emergency services.
    """

    chunks = chunk_by_section(text)
    print(chunks)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Call for Help"
