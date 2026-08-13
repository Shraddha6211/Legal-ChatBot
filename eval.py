"""
eval_retrieval.py
------------------
Easiest end-to-end retrieval eval for a RAG system.

Two commands:
    python eval_retrieval.py build-goldset   -> generates gold Q&A pairs from your chunks
    python eval_retrieval.py run             -> runs retrieval against the goldset and scores it

Metrics reported (deliberately simple, since each goldset question has exactly
one correct source chunk):
    - Hit Rate@k   : fraction of questions where the correct chunk appears
                     anywhere in the top-k retrieved results. This IS your
                     recall@k when there's only one relevant doc per query.
    - Precision@k  : hit / k for each question, averaged. (Only meaningful
                     with one relevant doc; with multiple relevant docs
                     you'd count how many of the k are relevant instead.)
    - MRR          : Mean Reciprocal Rank - rewards ranking the correct
                     chunk NEAR THE TOP, not just somewhere in top-k. This is
                     usually the most informative single number to watch.

Fill in `embed_texts()`, `retrieve()`, and `call_llm()` with your project's
actual embedding model, vector store, and LLM client. Everything else is
plug-and-play.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

GOLDSET_PATH = "goldset.jsonl"
RESULTS_PATH = "eval_results.json"


# ---------------------------------------------------------------------------
# Stable chunk IDs (content hash, not position) - makes the goldset survive
# re-chunking as long as the chunk text itself is unchanged.
# ---------------------------------------------------------------------------
def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# PROJECT IMPLEMENTATIONS
# ---------------------------------------------------------------------------
import chromadb
from openai import OpenAI
import config

# Initialize clients
chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def load_chunks() -> List[Dict[str, Any]]:
    """
    Load all chunks from ChromaDB collection.
    Returns a list of {"id": chunk_hash, "text": chunk_text}.
    """
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Fetch all documents from the collection
    all_docs = collection.get(include=["documents"])
    
    chunks = []
    if all_docs and all_docs["documents"]:
        for doc_text in all_docs["documents"]:
            chunk_id = chunk_hash(doc_text)
            chunks.append({
                "id": chunk_id,
                "text": doc_text
            })
    
    return chunks


def call_llm(prompt: str) -> str:
    """Call OpenAI's chat model to generate a question from a chunk."""
    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def retrieve(query: str, top_k: int) -> List[str]:
    """
    Retrieve top_k chunks from ChromaDB for a query.
    Returns a list of chunk hashes (IDs), ranked best-first.
    """
    # Get embedding for the query
    embedding_response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Query ChromaDB
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"]
    )
    
    # Convert retrieved texts to hashes (same hashing scheme as chunk_hash())
    retrieved_hashes = []
    if results and results["documents"] and results["documents"][0]:
        for doc_text in results["documents"][0]:
            retrieved_hashes.append(chunk_hash(doc_text))
    
    return retrieved_hashes


# ---------------------------------------------------------------------------
# Step 1: build a goldset from your own knowledge base
# ---------------------------------------------------------------------------
GOLDSET_GEN_PROMPT = """Here is a passage from a document:

---
{chunk_text}
---

Write ONE specific question that a user would realistically ask, which this \
passage directly and fully answers. Do not reference "the passage" in the \
question. Return only the question, nothing else."""


def build_goldset(sample_size: int = 40) -> None:
    """
    Generates {question, gold_chunk_id, gold_chunk_text} triples and writes
    them to a .jsonl file, one per line. Sample a subset of chunks rather
    than using every chunk - 30-50 good examples is plenty to start with,
    and it's much cheaper to hand-review than a full corpus dump.
    """
    chunks = load_chunks()
    if len(chunks) > sample_size:
        import random
        random.seed(42)  # reproducible sample
        chunks = random.sample(chunks, sample_size)

    with open(GOLDSET_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            question = call_llm(GOLDSET_GEN_PROMPT.format(chunk_text=c["text"])).strip()
            row = {
                "question": question,
                "gold_chunk_id": c["id"],
                "gold_chunk_text": c["text"],
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote goldset to {GOLDSET_PATH}. "
          f"IMPORTANT: skim it by hand and delete/edit any bad rows before trusting results.")


# ---------------------------------------------------------------------------
# Step 2: run retrieval against the goldset and score it
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    question: str
    gold_chunk_id: str
    retrieved_ids: List[str]
    hit: bool
    rank: int  # 1-indexed position of the gold chunk in results, 0 if not found


def evaluate(top_k: int = 5) -> Dict[str, Any]:
    if not os.path.exists(GOLDSET_PATH):
        raise FileNotFoundError(f"No goldset found at {GOLDSET_PATH}. Run `build-goldset` first.")

    rows: List[EvalRow] = []
    with open(GOLDSET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            gold = json.loads(line)
            retrieved_ids = retrieve(gold["question"], top_k=top_k)

            rank = 0
            if gold["gold_chunk_id"] in retrieved_ids:
                rank = retrieved_ids.index(gold["gold_chunk_id"]) + 1  # 1-indexed

            rows.append(EvalRow(
                question=gold["question"],
                gold_chunk_id=gold["gold_chunk_id"],
                retrieved_ids=retrieved_ids,
                hit=rank > 0,
                rank=rank,
            ))

    n = len(rows)
    hit_rate = sum(r.hit for r in rows) / n
    precision_at_k = sum((1 / top_k if r.hit else 0.0) for r in rows) / n
    mrr = sum((1 / r.rank if r.hit else 0.0) for r in rows) / n

    summary = {
        "num_questions": n,
        "top_k": top_k,
        "hit_rate_at_k": round(hit_rate, 3),     # == recall@k for single-relevant-doc goldsets
        "precision_at_k": round(precision_at_k, 3),
        "mrr": round(mrr, 3),
        "misses": [asdict(r) for r in rows if not r.hit],  # inspect these by hand
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "misses"}, indent=2))
    print(f"\n{len(summary['misses'])} misses written to {RESULTS_PATH} - look at these first.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build-goldset", "run"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()

    if args.command == "build-goldset":
        build_goldset(sample_size=args.sample_size)
    else:
        evaluate(top_k=args.top_k)


# ---------------------------------------------------------------------------
# HELPER FUNCTION FOR INLINE METRICS (used by rag.py and app.py)
# ---------------------------------------------------------------------------
def calculate_retrieval_metrics(
    distances: List[float],
    k: int = 4,
    similarity_threshold: float = 0.5,
    chunk_ids: List[str] = None,
    user_feedback: Dict[str, bool] = None,
    total_relevant: int = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate metrics for inline display during queries.
    This is compatible with the metrics.py interface but uses eval infrastructure.
    
    Args:
        distances: List of cosine distances from retrieval
        k: Top-k value for metrics
        similarity_threshold: Distance threshold for relevance classification
        chunk_ids: Optional chunk identifiers
        user_feedback: Optional user-marked relevance feedback
        total_relevant: Total relevant documents in corpus
    
    Returns:
        Dictionary of metric values (precision_at_k, relevant_count, etc.)
    """
    k = min(k, len(distances)) if distances else 0
    distances = distances[:k] if distances else []
    
    if chunk_ids is None:
        chunk_ids = [f"chunk_{i}" for i in range(len(distances))]
    else:
        chunk_ids = chunk_ids[:k]
    
    # Count relevant documents in top-k using similarity threshold
    relevant_count = 0
    for distance in distances:
        # User feedback takes priority (if available)
        is_relevant = distance < similarity_threshold
        
        if is_relevant:
            relevant_count += 1
    
    # Calculate precision@k
    precision_at_k = relevant_count / k if k > 0 else 0.0
    
    # Calculate recall@k (if total_relevant is known)
    recall_at_k = None
    if total_relevant and total_relevant > 0:
        recall_at_k = relevant_count / total_relevant
    
    # Calculate mean distance
    import numpy as np
    mean_distance = float(np.mean(distances)) if distances else 0.0
    min_distance = float(np.min(distances)) if distances else 0.0
    max_distance = float(np.max(distances)) if distances else 0.0
    
    metrics = {
        "precision_at_k": round(precision_at_k, 3),
        "recall_at_k": round(recall_at_k, 3) if recall_at_k is not None else None,
        "relevant_count": relevant_count,
        "k": k,
        "mean_distance": round(mean_distance, 4),
        "min_distance": round(min_distance, 4),
        "max_distance": round(max_distance, 4),
    }
    
    return metrics