# Financial Documents RAG

A command-line retrieval-augmented generation (RAG) application for asking
questions about central-bank and economic-statistics PDFs. It extracts
documents into searchable chunks, retrieves relevant passages using both
semantic and keyword search, reranks the results, and asks Gemini to answer
from the retrieved context.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini API key and model name
- A local model runtime that serves the Nomic embedding endpoint expected by
  the application at `http://localhost:12434`

The supplied [Docker Compose file](./docker-compose.yaml) declares Qwen3 and
Nomic embedding models for model-enabled Compose environments. The Python
application currently uses Gemini for text generation; embeddings are requested
from the local runtime.

## Setup

From the project root, install the dependencies:

```powershell
uv sync
```

Create a `.env` file in the project root with your Gemini and embedding
settings:

```dotenv
GEMINI_MODEL=your-gemini-model
GEMINI_API_KEY=your-gemini-api-key
EMBEDDING_API_URL=http://localhost:12434/engines/llama.cpp/v1/embeddings
EMBEDDING_MODEL_NAME=ai/nomic-embed-text-v1.5
```

The `.env` file is ignored by Git. Keep credentials there and do not commit
them. Set `EMBEDDING_API_URL` to the embedding endpoint exposed by your local
model runtime and `EMBEDDING_MODEL_NAME` to the embedding model identifier
served at that endpoint.

## Add and index documents

Place PDFs in `data/` (or another location you can provide). Run the ingestion
script and enter the PDF path when prompted:

```powershell
uv run python ingestion.py
```

The script extracts text and HTML tables, combines isolated headings with
following tables, splits oversized chunks, generates embeddings, and saves the
chunks to the local ChromaDB collection. The database is stored in
`chroma_db/`.

## Ask a question

Start the interactive command-line application:

```powershell
uv run python main.py
```

Enter a question at the prompt. The application expands it into additional
search queries, retrieves chunks in parallel, removes duplicate chunks,
reranks up to 15 retrieved candidates, selects up to eight, and sends the
selected context to Gemini. The answer is printed in the terminal.

## Architecture

```text
PDF
 └─ ingestion.py
     ├─ PyMuPDF4LLM extraction and chunking
     ├─ Local Nomic embedding endpoint
     └─ Persistent ChromaDB collection

Question
 └─ main.py
     ├─ augmentation.py ── Gemini query expansion
     ├─ retrieval.py
     │   ├─ ChromaDB dense search
     │   ├─ In-project BM25 keyword search
     │   ├─ Reciprocal-rank fusion
     │   └─ Related table-chunk expansion
     ├─ reranker.py ── Sentence Transformers CrossEncoder
     └─ Gemini answer generation with prompts from system_prompt.py
```

The code is organized as Python modules in the repository root rather than a
fully packaged application. `src/rag/__init__.py` is currently a placeholder;
use `python main.py` as shown above rather than the installed `rag` command.

## Configuration and implementation notes

- `GEMINI_MODEL` and `GEMINI_API_KEY` configure Gemini through LangChain.
- `EMBEDDING_API_URL` and `EMBEDDING_MODEL_NAME` configure the embedding
  endpoint and model used by `config.py`.
- ChromaDB persists data under `chroma_db/` in the project directory, in the
  `financial_pdfs` collection.
- `ingestion.py` uses a character-based token estimate when splitting chunks;
  it is not an exact tokenizer count.
- `system_prompt.py` requests citation markers and source references. The
  current application passes retrieved chunk dictionaries to Gemini without
  assigning citation numbers itself, so citation formatting depends on the
  content returned by retrieval.
- No automated test suite is currently included in the repository.