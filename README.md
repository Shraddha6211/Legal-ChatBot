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

# v1.2.0 – Chunking Strategy Experiments

This release focused entirely on improving the document chunking strategy while keeping the retrieval pipeline unchanged (ChromaDB + OpenAI embeddings).

## Changes

- Replaced the previous fixed-size chunking approach with a manual two-stage chunking pipeline:
  1. **Markdown Header Splitter**
  2. **Recursive Text Splitter**
- Implemented the pipeline entirely in Python without external RAG frameworks.
- Initial chunking configuration:
  - **Maximum chunk size:** 1000 tokens
  - **Chunk overlap:** 200 tokens

## Experiment 1

Using the above configuration produced approximately **1,800 chunks**.

### Observations

- Many chunks were extremely small, with some containing as few as **10 tokens**.
- The largest chunks contained around **1,290 tokens**.
- Because of the large number of chunks, generating embeddings became significantly slower than expected.

## Experiment 2

To reduce the number of embeddings and increase chunk density, the chunk size was increased.

### Updated configuration

- **Maximum chunk size:** 2000 tokens
- **Chunk overlap:** 400 tokens

### Results

- Total number of chunks decreased to approximately **900**.
- Embedding generation became more manageable.
- However, the dataset still contained many small chunks, with some containing only **50 tokens**, indicating that the splitter was still producing fragmented sections.

## Retrieval Experiments

Since the retrieved context sometimes appeared incomplete, different retrieval depths were tested.

### top_k = 15

the increasing retrieval depth to **15 chunks** produced slightly better context retrieval.

the LLM frequently responded that it could not find the requested information, even when the relevant answer was clearly present within the retrieved chunks.

### top_k = 30
the increasing **top_k** further to **30** did not improve performance.
Instead:
- More irrelevant chunks were retrieved.
- Context quality became noisier.
- Overall answer quality degraded.
This suggests that simply retrieving more chunks is not sufficient when the chunk boundaries themselves are suboptimal.

## Analysis
the primary bottleneck now appears to be the **Markdown Header Splitter** rather than the retrieval process.
the current Markdown document was generated automatically from a PDF and is not consistently structured. Since header-based splitting relies heavily on well-defined Markdown headings, poorly formatted or inconsistent headers likely resulted in:
* Excessive fragmentation,
* Numerous small chunks,
* Weak semantic grouping, and 
and reduced retrieval quality.

## Next Steps (v1.3.0)
the next release will focus on improving the source document rather than changing retrieval.
Planned improvements include:
* Re-convert the Constitution PDF into a cleaner, properly structured Markdown document.
* Re-run the header-based and recursive chunking pipeline on the improved Markdown.
* Re-evaluate chunk size distribution and retrieval quality before experimenting with further retrieval optimizations or reranking techniques.

## Current Conclusion
the retrieval pipeline itself appears to be functioning correctly. The experiments suggest that the limiting factor is the quality and structure of the source Markdown document. Improving the document structure is expected to produce more coherent chunks, better embeddings, and higher retrieval accuracy than simply increasing retrieval parameters such as `top_k`.
