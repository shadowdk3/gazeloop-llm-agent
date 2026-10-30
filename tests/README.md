# Test guide

The project uses `pytest`. The tests cover the RAG embedding, retrieval, and prompt-building path, plus basic application smoke checks. They do not test model quality or assert that a generated safety recommendation is correct.

## Test files

| Test file | What it checks | External services |
|---|---|---|
| `test_smoke.py` | Parses Python source files under the repository root and `src`, checks that `requirements.txt` is non-empty, imports selected third-party packages, and imports selected application modules. | No Ollama or PostgreSQL calls |
| `rag/test_chunking.py` | Splits a Markdown knowledge document into sections using level-2 headings (##). Checks that section titles and content are extracted correctly. | None |

There is currently no dedicated test for document ingestion, ranking quality, database error handling, or the final answer produced by the language model.

## Prerequisites for the full suite

Install project requirements:

```bash
python -m pip install -r requirements.txt
```

The embedding and RAG prompt/retrieval tests call the Ollama service using `nomic-embed-text`. Start Ollama and fetch that model before running those tests:

```bash
ollama pull nomic-embed-text
```

The retrieval and prompt tests also expect a PostgreSQL database with pgvector enabled, the `knowledge_chunks` table created, and relevant knowledge ingested. Follow the setup and ingestion instructions in [the RAG guide](../src/rag/readme.md).

The database connection uses these environment variable names: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `PGPASSWORD`. Configure them in the shell or test environment; the test code does not load a `.env` file itself.

## Running tests

- Run all tests from the repository root:

```
python -m pytest
```

- Run only the smoke checks, without needing the RAG services:

```
python -m pytest tests/test_smoke.py -q
```

- Run the chunking test

```
python -m pytest tests/rag/test_chunking.py -v
```

## Interpreting failures

- Connection errors from an embedding test usually mean Ollama is unavailable or the model has not been downloaded.
- Retrieval and prompt tests can fail if PostgreSQL is unavailable, the connection variables are incorrect, pgvector/the table is missing, or no chunks have been ingested.
- A non-empty-result assertion checks only that a result was returned. It does not measure whether the result is relevant or whether the recommendation is safe.
- The CI workflow currently runs `python -m pytest tests/ -q`; it therefore needs the external RAG services and data unless those tests are later changed to use mocks or service containers.
