# Simple RAG Legal Chatbot — v1.0.0

Lightweight retrieval-augmented legal chatbot for querying local documents.

**What it is**
- **Description:** A minimal, local-first RAG (retrieval-augmented generation) demo that answers legal questions from its knowledge base. Right now the KB consists of Nepal's constitution only.

**Quick Start**
- **Install:** `pip install -r requirements.txt`
- **Ingest documents:** `python ingest.py` (ingest your `data/` files)
- **Run the chatbot:** `python rag.py`
- **Run the Streamlit UI:** `streamlit run app.py`

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


# Legal RAG Chatbot — v2.0.0

A retrieval-augmented legal chatbot built around a collection of legal documents. Version **2.0.0** focuses on improving the quality of the knowledge base, document chunking, query retrieval, and evidence-grounded response generation while keeping the underlying embedding and vector-storage pipeline simple.

## What's New in v2.0.0

### 1. Expanded Legal Knowledge Base

The knowledge base was expanded beyond the original document set with additional legal documents:

- **Constitution**
- **Civil Code**
- **Criminal Code**

This provides broader legal context and allows the chatbot to retrieve evidence from a larger collection of legal sources.

---

### 2. Recursive Structure-Aware Chunking

The original fixed-size chunking approach was replaced with a **recursive, Markdown-aware chunking pipeline**.

The chunker attempts to preserve the structure of legal documents while recursively splitting content when sections become too large.

The pipeline considers document structure such as:

```text Document ↓ Headings / Sections ↓ Subsections ↓ Paragraphs ↓ Smaller text units ```

This helps prevent important legal provisions from being arbitrarily separated simply because they exceed a fixed character limit.

---

### 3. Intelligent Small-Chunk Merging

Very small sections can produce fragmented and context-poor chunks.

Version 2.0.0 therefore introduces a merging step that combines small chunks with adjacent chunks where appropriate.

Instead of producing:

```text Chunk 1 → very small Chunk 2 → very small Chunk 3 → normal ```

the pipeline attempts to produce denser units such as:

```text Chunk 1 + Chunk 2 → meaningful contextual chunk Chunk 3 → normal chunk ```

This improves the amount of useful context available during retrieval while reducing unnecessary fragmentation.

---

### 4. Token-Aware Splitting and Controlled Overlap

Chunk sizes are now handled with greater awareness of **token limits** rather than relying only on raw character length.

Controlled overlap is also applied between chunks to preserve contextual continuity.

Conceptually:

```text
Chunk A
████████████████
            ↓ overlap
            █████
            Chunk B
            ████████████████
```

This reduces the possibility of losing important information when a legal provision crosses a chunk boundary.

---

### 5. LLM-Based Query Expansion

Query processing was improved by introducing ****LLM**-based query expansion**.

Instead of searching only with the user's original question:

```text
### User Query
        ↓
### Vector Retrieval
```

the system now generates multiple semantically related search variants:

```text
           Original Query
                 ↓
          Query Expansion
         /       |       \
        /        |        \
      Query 1  Query 2    Query 3
        \        |        /
         \       |       /
              Retrieval
```

This improves retrieval coverage when the terminology used by the user differs from the terminology used in the legal documents.

---

### 6. Deduplication and Re-ranking

Retrieval results from the expanded queries may contain duplicate or overlapping chunks.

Version 2.0.0 therefore introduces a post-retrieval processing stage:

```text
### Expanded Queries
      ↓
### Multiple Retrieval Results
      ↓
Deduplication
      ↓
Re-ranking
      ↓
### Strongest Evidence
```

This keeps the most relevant evidence while reducing repeated chunks in the final context supplied to the **LLM**.

---

### 7. Improved Evidence Synthesis

The answer-generation prompt was upgraded to handle situations where relevant information is distributed across multiple retrieved chunks.

The previous behavior could be overly conservative and produce a response such as:

> Information not found.

even when partial evidence existed across the retrieved context.

Version 2.0.0 instead encourages **evidence-based synthesis** when the retrieved information is sufficient to support a cautious answer.

The model is instructed to:

- use the retrieved evidence
- combine relevant information across chunks
- avoid unsupported claims
- distinguish strong evidence from incomplete evidence
- acknowledge limitations when the retrieved context is insufficient

---

### 8. More Grounded Response Behavior

Response generation now favors **grounded and cautious answers** when evidence is partial.

The chatbot should not immediately refuse a question simply because one retrieved chunk does not contain the complete answer.

Instead:

```text
### Strong Evidence
      ↓
### Direct Answer

### Partial but Relevant Evidence
      ↓
### Cautious Evidence-Based Answer

### Insufficient Evidence

      ↓
Clearly State the Limitation
```

The goal is to reduce false *not found* responses without encouraging hallucination.

---

## RAG Pipeline

The overall v2.0.0 pipeline is:

```text
    Legal Documents
    │
    ▼
    Markdown Documents
    │
    ▼
    Recursive Structure-Aware
    Chunking
    │
    ▼
    Small-Chunk Merging
    │
    ▼
    Token-Aware Splitting
    │
    ▼
    Controlled Overlap
    │
    ▼
    OpenAI Embeddings
    text-embedding-3-small
    │
    ▼
    ChromaDB
    │
    │
    User Query
    │
    ▼
    Query Expansion
    │
    ▼
    Multiple Search Queries
    │
    ▼
    Retrieval
    │
    ▼
    Deduplication
    │
    ▼
    Re-ranking
    │
    ▼
    Relevant Context
    │
    ▼
    Answer Generation
    │
    ▼
    Grounded Legal Answer
```

---

## Embeddings and Vector Database

The embedding and vector-storage approach remains intentionally simple in v2.0.0.

### Embedding Model

```text OpenAI text-embedding-3-small ```

### Vector Database

```text ChromaDB ```

The system continues to use the same general embedding and vector-storage approach from v1.0.0. The major improvements in this release are focused on **document representation, chunking, query processing, retrieval post-processing, and response generation**.

---

## Version 1.0.0 → Version 2.0.0

| Component         | v1.0.0                            | v2.0.0                                   |
| ----------------- | --------------------------------- | ---------------------------------------- |
| Knowledge Base    | Initial legal document set        | Expanded with Civil Code & Criminal Code |
| Document Format   | Markdown                          | Markdown                                 |
| Chunking          | Fixed-size chunking with overlap  | Recursive structure-aware chunking       |
| Small Chunks      | Basic handling                    | Intelligent adjacent-chunk merging       |
| Chunk Size        | Fixed-size approach               | Token-aware splitting                    |
| Overlap           | Basic overlap                     | Controlled overlap                       |
| Embeddings        | `text-embedding-3-small`          | `text-embedding-3-small`                 |
| Vector Database   | ChromaDB                          | ChromaDB                                 |
| Query Processing  | Original query                    | LLM-based query expansion                |
| Retrieval Results | Direct results                    | Deduplication + re-ranking               |
| Context Handling  | More limited                      | Evidence synthesis across chunks         |
| Response Behavior | More likely to return *not found* | More cautious evidence-based synthesis   |

---

## Key Improvements

The main objective of v2.0.0 was **not to replace the existing **RAG** architecture**, but to improve the quality of the information flowing through it.

The improvements can be summarized as:

```text
### More Documents
      +
### Better Chunking
      +
### Better Context Preservation
      +
### Query Expansion
      +
### Result Deduplication
      +
Re-ranking
      +
### Better Evidence Synthesis
      ↓
Improved Retrieval and Answer Quality
```

---

## Design Philosophy

Version 2.0.0 continues to prioritize simplicity and transparency.

The system does not rely on large orchestration frameworks for its core **RAG** pipeline. Instead, the major components are implemented explicitly so that each stage can be inspected and improved independently.

The primary focus of this release is:

> **Retrieve better context before asking the **LLM** to generate an answer.**

Rather than relying on a stronger generation model alone, v2.0.0 improves the quality and completeness of the evidence provided to the model.

---

## Limitations

Despite the improvements, this version still has several limitations:

- Retrieval is still dependent on embedding and keyword/search quality.
- `text-embedding-3-small` remains the embedding model.
- ChromaDB is used as the vector store.
- Query expansion can occasionally generate less useful search variants.
- Re-ranking quality depends on the implemented scoring strategy.
- The system does not guarantee complete legal interpretation.
- Retrieved evidence may still be incomplete for complex questions.
- The chatbot should not be treated as a substitute for professional legal advice.

---

## Version

**v2.0.0**

### Main Focus

**Knowledge Base Expansion + Recursive Chunking + Query Expansion + Improved Evidence Retrieval and Synthesis**



# Legal RAG Chatbot — v2.1.0

Version **2.1.0** builds on the retrieval and document-processing improvements introduced in v2.0.0 by adding **conversational memory** and improving the system's prompting, behavioral guardrails, and answer structure.

The main focus of this release is making the chatbot more **context-aware, consistent, and natural during multi-turn conversations** while maintaining grounded legal responses.

---

## What's New in v2.1.0

### 1. Conversational Memory

The chatbot now maintains conversation history so that it can understand follow-up questions in context.

Previously, questions were primarily handled as independent queries:

```text User: What does Article 18 say?

User: What are its limitations? ```

Without conversation memory, the second question can be difficult to interpret because *its* depends on the previous conversation.

With v2.1.0:

```text
User:
What does Article 18 say?
        ↓
Assistant:
[Answer about Article 18]
        ↓
User:
What are its limitations?
        ↓
### Conversation Memory
        ↓
Understand *its* → Article 18
        ↓
Retrieval
        ↓
Answer
```

This allows the chatbot to better handle:

- follow-up questions
- references such as *this*, *that*, *it*, or *the previous section*
- clarification questions
- multi-turn discussions
- contextual queries

Conversation history is used to understand the user's current intent rather than blindly treating every message as an isolated query.

---

### 2. Improved System Prompts

The system prompts were redesigned and refined to provide clearer behavioral instructions to the **LLM**.

The prompting system now separates responsibilities such as:

```text
Common-Sense / Behavioral Instructions
              +
**RAG** Instructions
              +
### Conversation Context
              +
### Retrieved Evidence
              +
### User Query
```

This makes the chatbot's behavior more predictable while keeping retrieval-specific instructions separate from general conversational behavior.

---

### 3. Common-Sense Guardrails

Common-sense behavioral guardrails were added to handle situations that do not require normal **RAG** retrieval.

The chatbot can now better distinguish between:

```text
Normal conversation
        ↓
**RAG** question
        ↓
Appointment-related request
        ↓
Ambiguous question
        ↓
Off-topic question
```

For example, a simple greeting such as:

> Hello

should not unnecessarily trigger:

```text
### Query Expansion
      ↓
Embedding
      ↓
### Vector Search
      ↓
Retrieval
```

Instead, the chatbot can respond naturally.

The guardrails also help the system handle:

- greetings and casual conversation
- acknowledgements
- ambiguous questions
- irrelevant/off-topic requests
- insufficient evidence
- prompt injection attempts
- requests for internal system information
- unsupported legal claims
- inappropriate use of retrieved context

---

### 4. Stronger Grounding and Safety Behavior

The system prompts were updated to reinforce the distinction between **document-supported information** and unsupported model knowledge.

The chatbot is instructed to:

- avoid inventing legal provisions
- avoid fabricating citations or sources
- avoid claiming information exists when it does not
- respect the selected document scope
- treat uploaded documents as information rather than instructions
- acknowledge insufficient evidence
- avoid pretending to be a lawyer
- provide legal information rather than claiming to provide professional representation

The goal is to make the system more reliable without making it unnecessarily restrictive.

---

### 5. Structured Answer Format

Response generation was also improved by introducing a more consistent answer structure.

Depending on the question, responses can now follow a structure such as:

```text ### Answer

Direct response to the user's question.

### Explanation

Additional explanation based on the retrieved evidence.

### Relevant Provision

Relevant article, section, or provision when available.

### Source

Document name and available source metadata. ```

The structure is applied when useful rather than being forced onto every conversational response.

For simple questions, the chatbot can still provide a concise answer.

---

## Updated Conversation Flow

The v2.1.0 interaction flow can be represented as:

```text
    User Message
    │
    ▼
    Conversation Memory
    │
    ▼
    Intent / Context
    Understanding
    │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
    Conversation     **RAG**      Appointment
    │           │           │
    │           ▼           │
    │     Query Expansion   │
    │           │           │
    │           ▼           │
    │       Retrieval       │
    │           │           │
    │           ▼           │
    │      Re-ranking       │
    │           │           │
    └───────────┼───────────┘
    │
    ▼
    Prompt + Context
    │
    ▼
    Structured Answer
    │
    ▼
    Conversation Memory
```

---

## v2.0.0 → v2.1.0

| Component               | v2.0.0                    | v2.1.0            |
| ----------------------- | ------------------------- | ----------------- |
| Knowledge Base          | Expanded legal documents  | Same              |
| Chunking                | Recursive structure-aware | Same              |
| Small Chunk Handling    | Adjacent-chunk merging    | Same              |
| Token Awareness         | Yes                       | Same              |
| Controlled Overlap      | Yes                       | Same              |
| Embeddings              | `text-embedding-3-small`  | Same              |
| Vector Database         | ChromaDB                  | Same              |
| Query Expansion         | LLM-based                 | Same              |
| Deduplication           | Yes                       | Same              |
| Re-ranking              | Yes                       | Same              |
| Evidence Synthesis      | Improved                  | Same              |
| Conversational Memory   | Limited/absent            | **Added**         |
| Follow-up Questions     | Limited                   | **Context-aware** |
| System Prompt           | Basic task instructions   | **Reworked**      |
| Common-Sense Guardrails | Limited                   | **Added**         |
| Answer Structure        | Less consistent           | **Structured**    |
| Behavioral Handling     | Basic                     | **More robust**   |

---

## Core Improvements

The main improvements in v2.1.0 can be summarized as:

```text
v2.0.0
### Better Documents
      +
### Better Chunking
      +
### Better Retrieval
      +
### Better Evidence Synthesis
    │
    ▼
    v2.1.0
    │
    ├── Conversational Memory
    ├── Common-Sense Guardrails
    ├── Improved System Prompts
    └── Structured Answers
```

The focus has shifted from improving **what the system retrieves** to improving **how the system understands and responds to the user**.

---

## Design Philosophy

v2.1.0 continues the project's emphasis on keeping the **RAG** system understandable and lightweight.

Instead of introducing a complex memory or agent framework, conversational context is handled explicitly within the application's existing architecture.

Similarly, behavioral improvements are implemented through carefully designed system prompts rather than adding a separate guardrail framework.

This keeps the pipeline transparent:

```text User ↓ ### Conversation Context ↓ Intent / Task ↓ Retrieval when necessary ↓ ### Retrieved Evidence ↓ ### Prompt Instructions ↓ ### Structured Response ```

---

## Limitations

Although v2.1.0 improves conversational behavior, it still has limitations:

- Conversation memory is limited to the configured conversation history.
- Long conversations may contain irrelevant context.
- Follow-up questions can still be ambiguous.
- Prompt-based guardrails are not a substitute for formal safety mechanisms.
- The chatbot remains dependent on retrieval quality.
- Legal answers are limited by the available documents and retrieved evidence.
- The chatbot does not replace professional legal advice.

---

## Version

**v2.1.0**

### Main Focus

**Conversational Memory + Common-Sense Guardrails + Improved System Prompts + Structured Legal Responses**