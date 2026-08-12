# Simple RAG Legal Chatbot

A lightweight legal chatbot built with OpenAI, ChromaDB, and Streamlit. This project demonstrates a minimal retrieval-augmented generation (RAG) pipeline for legal query answering, plus a single-document upload/query workflow and a placeholder appointment-booking experience. Handles queries and pdfs in both English and Nepali language and responds accordingly.

## Project Overview

This repo is organized to keep the core RAG pipeline separate from the Streamlit UI and uploaded-document support.

### What it does
- Answers legal questions from a local knowledge base using RAG.
- Supports a dedicated document upload mode that answers questions from a single uploaded PDF only.
- Provides a booking experience placeholder using an embedded Calendly iframe.

### Core features
- **Knowledge base search**: query expansion, retrieval from ChromaDB, and answer generation via OpenAI chat.
- **Single-document mode**: upload a PDF, ingest it into ChromaDB, and restrict Q&A strictly to that document using metadata filters.
- **Document summarization**: summarize uploaded documents using chunk-based summarization.
- **Interactive UI**: Streamlit dashboard with three experience options:
  1. Ask a Legal Query
  2. Upload & Query a Document
  3. Book an Appointment

---

## Repository Structure

- `app.py` — Streamlit dashboard and experience orchestration.
- `rag.py` — query-time logic, retrieval, prompt creation, and answer generation.
- `document.py` — PDF ingestion, Markdown conversion, chunk creation, and document-specific ChromaDB operations.
- `chunking.py` — recursive Markdown-aware chunking and token-aware split/merge logic.
- `ingest.py` — CLI helper to read local markdown and build the vector database.
- `config.py` — central configuration for API keys, paths, model names, and collection settings.
- `requirements.txt` — Python dependencies for OpenAI, ChromaDB, Streamlit, PDF parsing, and tokenization.
- `data/` — sample documents for ingestion.
- `chroma_db/` — persistent ChromaDB storage.
- `README_earlier.md` — earlier project notes, experiments, and version history.

---

## Detailed file responsibilities

### `app.py`
- Uses `st.session_state` to manage UI mode and conversation history.
- Presents three main experiences in the dashboard.
- Renders the booking page with a responsive Calendly iframe.
- Handles document upload, summary generation, and document-only Q&A.
- Uses compact inline buttons to keep the document action controls clean.

### `rag.py`
- Connects to ChromaDB and OpenAI.
- Expands user queries into multiple search variants.
- Rewrites follow-on questions using conversation history.
- Retrieves and deduplicates vectors from ChromaDB.
- Builds grounded prompts for response generation.
- Supports `metadata_filter` to restrict retrieval to a single uploaded document.

### `document.py`
- Converts uploaded PDFs into normalized Markdown.
- Builds document chunks with metadata and document-scoped IDs.
- Embeds uploaded document chunks in ChromaDB.
- Deletes document-specific vectors when an uploaded document is replaced.

### `chunking.py`
- Implements token-aware chunking with a Markdown-first strategy.
- Splits by headings, merges small sections, and recursively splits large sections.
- Falls back to paragraph, sentence, and token-level splitting when needed.
- Produces dense chunks with overlap metadata for better retrieval quality.

### `ingest.py`
- Reads the configured markdown document.
- Builds chunks using `chunking.py`.
- Embeds them into ChromaDB with metadata.
- Useful for bootstrapping the initial knowledge base from local documents.

### `config.py`
- Loads environment variables from `.env`.
- Stores OpenAI key, ChromaDB path, collection name, and model settings.

---

## What changed in this project

### Initial baseline
- A simple RAG pipeline that ingested a local markdown file and answered legal questions.
- Used ChromaDB for vector storage and OpenAI embeddings/chat models for retrieval and response generation.

### New improvements
- Added a Streamlit dashboard with a polished experience selector.
- Added a third view: **Book an Appointment**.
- Added single-document upload support:
  - Upload a PDF.
  - Convert it to Markdown.
  - Create document-specific chunks with metadata.
  - Ingest those chunks into ChromaDB.
  - Restrict later retrieval to `document_id == active_document_id`.
- Added inline document action buttons:
  - `Generate Summary`
  - `Ask Questions`
  - `Remove Document`
- Improved UI layout so the three document buttons render horizontally with minimal spacing.

### Behavior guarantees
- Only one uploaded document is active at a time.
- Uploading a new document deletes the previous document's vectors and clears its chat state.
- Document-mode answers are generated only from document chunks via a strict metadata filter.
- Summaries are generated from the uploaded document's chunks, not from external data.

---

## Setup & usage

1. Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit UI:

```bash
streamlit run app.py
```

4. Choose one of the three experiences:

- `Ask a Legal Query` — search the existing knowledge base.
- `Upload & Query a Document` — upload a PDF and ask document-only questions.
- `Book an Appointment` — view the placeholder Calendly iframe.

---

## Notes for developers

- The project avoids external orchestration frameworks like LangChain or LlamaIndex.
- All ingestion, retrieval, and summarization logic is implemented in plain Python.
- The vector store is persisted locally in `chroma_db/` so the project can be run offline after initial ingestion.
- The document upload mode uses `openai` embeddings directly and writes metadata for clean isolation.

## Recommended next steps

- Replace the Calendly placeholder link with a real scheduling URL.
- Add automated tests for document ingestion, metadata filtering, and Streamlit mode switching.
- Improve PDF extraction to preserve headings and formatting more accurately.
- Add client-side validation for uploaded PDFs and better error handling.

---

## License

See the `LICENSE` file for licensing details.
