from src.rag.prompt import build_safety_prompt
from src.rag.models import RetrievedChunk

def test_rag_prompt():
    """Test prompt generation using predefined guideline chunks."""

    # Simulate a visual description from the vision model.
    visual_description = (
        "A person appears to be lying on the ground "
        "on an icy outdoor surface."
    )

    # Simulate guideline chunks returned by the retriever.
    # These are predefined test data, not real database results.
    chunks = [
        RetrievedChunk(
            document_name="fall_response.md",
            section="1. Incident Verification",
            content="Verify whether a person has fallen.",
            similarity=0.82,
        ),
        RetrievedChunk(
            document_name="fall_response.md",
            section="2. Immediate Response",
            content="Check whether the person is conscious.",
            similarity=0.91,
        ),
    ]

    # Build the prompt using the predefined test data.
    prompt = build_safety_prompt(
        visual_description,
        chunks,
    )
    
    print()
    print(prompt)

    # Verify that the prompt contains the expected information.
    assert visual_description in prompt
    assert "Relevant safety policy" in prompt
    assert "Verify whether a person has fallen." in prompt
    assert "Check whether the person is conscious." in prompt
    assert "fall_response.md" in prompt
    assert "1. Incident Verification" in prompt
    assert "2. Immediate Response" in prompt