from pathlib import Path
import os
import re

"""
Reading a Markdown knowledge document and splitting it into smaller sections (chunks). 
These chunks are later passed to the embedding service.
"""

# loading a Markdown knowledge document and splitting it into smaller, structured sections (chunks)
script_path = Path(__file__).resolve()
script_dir = script_path.parent
project_dir = script_dir.parent.parent

def load_document(file_path):
    """
    Read a Markdown document and return its contents as a string.
    """
    return Path(file_path).read_text(encoding="utf-8")

def chunk_by_section(text):
    """
    Split Markdown text into chunks using level-2 headings (##).

    Ignore the document title and introduction before the first section.
    Support headings with or without bold formatting.
    """

    pattern = r"^\s*(?:\*\*)?##\s+(.+?)(?:\*\*)?\s*$"
    
    # Split the document whenever a line starts with '## '
    sections = list(re.finditer(pattern, text, flags=re.MULTILINE))
    
    chunks = []

    for i, section in enumerate(sections):
        title = section.group(1).strip()

        start = section.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)

        content = text[start:end].strip()

        if content:
            chunks.append({
                "section": title,
                "content": content
            })

    return chunks

target_file = os.path.join(project_dir, "knowledge", "fall_response.md")

# Loads a Markdown document
text = load_document(target_file)

# Splits the document into sections based on ## headings
chunks = chunk_by_section(text)

# Creates structured chunks containing a section title and its content
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)