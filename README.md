# Simple RAG Legal Chatbot — v1.0.0

Lightweight retrieval-augmented legal chatbot for querying local documents.

**What it is**
- **Description:** A minimal, local-first RAG (retrieval-augmented generation) demo that answers legal questions from its knowledge base. Right now the KB consists of Nepal's constitution only.

**Quick Start**
- **Install:** `pip install -r requirements.txt`
- **Ingest documents:** `python ingest.py` (ingest your `data/` files)
- **Run the chatbot:** `python rag.py`

**Release**
- **This release:** v0.1 — initial public demo. To publish on GitHub, tag `v0.1` and create a Release with a short changelog.

**Phase 1 (Done ✅)**
- Fixed-size chunking
- Overlap
- ChromaDB
- Cosine similarity
- OpenAI embeddings

**License**
- See the `LICENSE` file for terms.

Thanks for trying this first release — feedback welcome.