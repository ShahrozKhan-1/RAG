# Financial Documents RAG

A retrieval-augmented generation (RAG) chatbot for questions about central-bank
and economic-statistics PDFs. The repository includes an interactive
Streamlit interface and a command-line version that use the same retrieval,
reranking, and Gemini generation pipeline.

## Requirements

- Python 3.12
- A Gemini API key and model name
- A local model runtime exposing the embedding endpoint configured below
- The Python dependencies in `requirements.txt`

## Setup on Windows

From the project directory, create and activate a virtual environment and
install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GEMINI_MODEL=your-gemini-model
GEMINI_API_KEY=your-gemini-api-key
EMBEDDING_API_URL=http://localhost:12434/engines/llama.cpp/v1/embeddings
EMBEDDING_MODEL_NAME=ai/nomic-embed-text-v1.5
```

`GEMINI_MODEL` and `GEMINI_API_KEY` configure text generation. The embedding
URL and model name are read from `EMBEDDING_API_URL` and
`EMBEDDING_MODEL_NAME`. Start the configured local model runtime before
ingesting documents or asking questions. The `.env` file is excluded from
version control; do not commit credentials.

The supplied [Docker Compose configuration](./docker-compose.yaml) declares
Qwen3 and Nomic models for compatible model-enabled Compose environments. The
Python application currently uses Gemini for generation and the configured
local service for embeddings.

## Index PDF documents

Place PDF files in `data/`, then run:

```powershell
python ingestion.py
```

Enter the PDF path when prompted. Ingestion extracts text and HTML tables with
PyMuPDF4LLM, joins isolated headings with tables, splits oversized chunks,
generates embeddings, and saves the document text, metadata, and vectors to
the persistent ChromaDB collection. ChromaDB data is stored in `chroma_db/`.

## Start the Streamlit chatbot

From the project root, run:

```powershell
streamlit run app.py
```

The app opens in a browser. Ask a question in the input at the bottom of the
page. Your conversation is kept in Streamlit session state for follow-up
questions; use **New chat** to clear it.

The interface calls the shared RAG functions in `main.py`. It expands the
question, retrieves candidates with dense and BM25 search, combines their
rankings, expands related table chunks, reranks the results with a Sentence
Transformers cross-encoder, and sends the selected context to Gemini.
Retrieved source metadata is available in the collapsible **Sources** section.

## Command-line interface

The original terminal workflow remains available:

```powershell
python main.py
```

## Project structure

```text
app.py             Streamlit chat interface and session state
styles.css         Responsive dark-teal interface styles
augmentation.py    Gemini-based search-query augmentation
config.py          Gemini, local embedding, and ChromaDB configuration
ingestion.py       PDF extraction, chunking, embedding, and storage
retrieval.py       Dense retrieval, BM25, rank fusion, and table expansion
reranker.py        Sentence Transformers cross-encoder reranking
system_prompt.py   Answering and query-augmentation prompts
main.py            Shared RAG pipeline and command-line application
data/              Example statistical-supplement PDF
chroma_db/         Local persistent vector database (generated)
```

The UI uses native Streamlit forms and chat-message widgets for interaction;
custom CSS in `styles.css` provides the visual design. The Streamlit UI does
not replace or modify the existing command-line entry point.

## Notes

- Chunk token sizes in `ingestion.py` are estimated from character counts,
  rather than measured with a model tokenizer.
- The answer prompt requests citations. `main.py` supplies numbered
  context markers and source labels derived from retrieved chunks; verify the
  cited passages in the displayed source metadata.
- No automated test suite is currently included.
