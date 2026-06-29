# Context-Aware Document Q&A Bot

A local Retrieval-Augmented Generation (RAG) app that answers questions using
only uploaded PDFs or pasted text. It is built with Streamlit, FAISS,
Sentence Transformers embeddings, LangChain, and a local Ollama model.

## Key features

- Upload PDF files or paste raw text.
- Clean, chunk, embed, and index documents for retrieval.
- Retrieve the most relevant source chunks with FAISS.
- Answer questions only from the provided document context.
- Return the exact refusal:

```text
This information is not available in the document.
```
- Display document statistics, source chunks, confidence, and chat history.

## Repository structure

- `app.py` — Streamlit UI and application flow.
- `rag_pipeline.py` — document ingestion, FAISS retrieval, and Ollama QA logic.
- `requirements.txt` — runtime dependencies.
- `requirements-dev.txt` — development and test dependencies.
- `pyproject.toml` — project metadata and tooling configuration.
- `tests/` — unit tests.
- `scripts/debug_rag.py` — optional local debug helper.
- `.github/workflows/python-app.yml` — GitHub Actions CI workflow.
- `.streamlit/config.toml` — Streamlit server configuration.
- `.gitignore` — files excluded from source control.
- `LICENSE` — MIT license.

## Prerequisites

- Python 3.11 or newer
- Ollama installed locally
- Enough disk space for the Ollama model and sentence-transformer cache

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ollama setup

Verify available models:

```bash
ollama list
```

Pull the recommended default model:

```bash
ollama pull llama3.2:3b
```

Start the Ollama server:

```bash
ollama serve
```

## Run the application

With the virtual environment active:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit, typically:

```text
http://localhost:8501
```

## Using the app

1. Upload PDFs or paste text in the sidebar.
2. Choose an Ollama model if needed.
3. Click **Process Document**.
4. Ask a question in the main area.
5. Review the answer, the confidence score, and retrieved context.

## Recommended default model

The app defaults to `llama3.2:3b`. The sidebar lets you override it per session.

- `tinyllama` — fastest, lowest resource usage, good for quick testing.
- `llama3.2:3b` — recommended default for a good quality/speed balance.
- `llama3:latest` / `mistral:latest` — higher quality but require more RAM and disk.

> Do not use embedding-only models such as `nomic-embed-text:latest` as the chat model.

## Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the unit tests:

```bash
pytest
```

## Optional local debugging

Use the helper script for quick manual verification:

```bash
python scripts/debug_rag.py
```

## Troubleshooting

- If the app cannot reach Ollama, confirm the server is running and the model is pulled.
- If a PDF has no extractable text, it may be scanned or image-only. Paste the text manually instead.
- If the first run is slow, the embedding model may still be downloading or caching.

