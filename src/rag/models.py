from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    document_name: str
    section: str
    content: str
    similarity: float