# rag.py
# This script handles query-time logic: given a user's question,
# expand it, retrieve relevant chunks, and generate an answer.

import re

import config
import chromadb
from openai import OpenAI

client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_embedding(text):
    """
    Sends a piece of text to OpenAI's embedding model and returns
    the resulting vector. (Same function as in ingest.py — we need
    the identical model here so question vectors and chunk vectors
    live in the same space.)
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_collection():
    """
    Connects to our existing persistent ChromaDB collection
    (the one ingest.py already populated).
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _clean_expanded_query_lines(text):
    """
    Turns the chat model's response into a list of plain query strings.
    We keep this line-based so the expansion step stays simple and does not
    depend on structured output.
    """
    lines = []

    for raw_line in text.splitlines():
        cleaned_line = raw_line.strip()

        if not cleaned_line:
            continue

        cleaned_line = re.sub(r"^\s*\d+[\).:-]?\s*", "", cleaned_line)
        cleaned_line = re.sub(r"^[-*•]+\s*", "", cleaned_line)

        if cleaned_line:
            lines.append(cleaned_line)

    return lines


def expand_query(question, max_expansions=4):
    """
    Uses the chat model's general knowledge to turn the user's question into
    a few alternative search queries.
    """
    prompt = f"""You expand user questions for semantic retrieval over a legal document.

Return up to {max_expansions} short search queries on separate lines.
Use synonyms, related legal terms, article names, abbreviations, and likely
wording variations that would help retrieve the right document chunks.

Rules:
- Output only the queries, one per line.
- Do not use numbering, bullets, labels, or explanations.
- Keep each line short and focused.

Question:
{question}

Expanded queries:"""

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    expanded_queries = _clean_expanded_query_lines(
        response.choices[0].message.content
    )

    unique_queries = []
    seen_queries = set()

    for query_text in expanded_queries:
        normalized_query = query_text.lower()

        if normalized_query in seen_queries:
            continue

        seen_queries.add(normalized_query)
        unique_queries.append(query_text)

    return unique_queries[:max_expansions]


def build_search_queries(question):
    """
    Creates the original query plus LLM-expanded variants.
    """
    search_queries = [question]

    for expanded_query in expand_query(question):
        if expanded_query.lower() not in [query.lower() for query in search_queries]:
            search_queries.append(expanded_query)

    return search_queries


def retrieve_relevant_chunks(question, top_k=4):
    """
    Given a user's question, returns the top_k most relevant chunks from the
    vector database after query expansion.
    """
    # Step 1: connect to the stored collection
    collection = get_collection()

    # Step 2: create alternate search phrasings using the model's knowledge
    search_queries = build_search_queries(question)

    # Step 3: query ChromaDB for each phrasing and keep the best results we see
    ranked_chunks = []

    for query_text in search_queries:
        query_embedding = get_embedding(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_texts = results["documents"][0]
        distances = results["distances"][0]

        for chunk_text, distance in zip(retrieved_texts, distances):
            found_index = None

            for index, (known_chunk_text, known_distance) in enumerate(ranked_chunks):
                if known_chunk_text == chunk_text:
                    found_index = index

                    if distance < known_distance:
                        ranked_chunks[index] = (known_chunk_text, distance)

                    break

            if found_index is None:
                ranked_chunks.append((chunk_text, distance))

    ranked_chunks.sort(key=lambda item: item[1])
    ranked_chunks = ranked_chunks[:top_k]

    retrieved_texts = [item[0] for item in ranked_chunks]
    distances = [item[1] for item in ranked_chunks]

    return retrieved_texts, distances, search_queries

def build_prompt(question, retrieved_chunks):
    """
    Combines the retrieved chunks and the user's question into a single
    prompt string, with clear instructions for the model to stay grounded
    in the provided context.
    """
    # Present each chunk as a separate evidence snippet so the model can
    # compare them more reliably instead of treating the context as one blob.
    context_lines = []

    for index, chunk_text in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"Snippet {index}: {chunk_text}")

    context_text = "\n\n".join(context_lines)

    prompt = f"""You are a legal information assistant for Nepal, answering questions using the retrieved context below as your primary source of truth.

## Grounding Policy (moderate strictness)
- Base your answer on the context provided. Do not pull in outside legal facts, section numbers, or provisions that are not present in the context.
- You do NOT need a verbatim match. If the answer is implied, paraphrased, or can be reasonably synthesized by combining multiple snippets, do so and answer directly.
- Weak or partial evidence is still evidence — give the best-supported answer and briefly flag what's uncertain (e.g., "the context suggests X, but doesn't specify Y").
- Only respond with "I couldn't find that information in the provided document." if NONE of the retrieved snippets are relevant to the question at all. This should be rare — check the context carefully before concluding this.
- Never invent a specific section, article, or clause number that isn't in the context. If the context names the relevant Act/Code but not the exact section, say so explicitly (e.g., "under the relevant provisions of the Muluki Civil Code, 2074, though the exact section is not specified in the retrieved context") rather than guessing a number or refusing outright.

## Response Style
First, decide whether the question is:
- **Factual** — asks for a definition, date, right, rule, or direct legal fact.
- **Scenario-based** — a hypothetical, case study, or "what happens if..." situation involving multiple facts.

**For Factual questions:**
- Lead with the direct answer in the first sentence.
- No filler ("Sure, I can help with that...").
- Name the specific law/section/article if the context provides it.

**For Scenario-based questions**, structure your answer with these sections:
- **Relevant facts & governing law** — what law/provision applies, and the baseline legal position, based on the context.
- **Key issue** — the core factor that determines the outcome (1 short paragraph).
- **What this means for the parties** — a short, numbered list of practical implications or steps, grounded in the context.
- **Caveats** — anything the context doesn't cover, stated plainly (not hedged into a refusal).

Use markdown headers and bullets for scenario answers; keep factual answers short and unstructured.

Context:
{context_text}

Question:
{question}

Answer:"""
    return prompt


def generate_answer(question, retrieved_chunks):
    """
    Sends the constructed prompt to OpenAI's chat model and returns
    the model's answer as a string.
    """
    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the text of the model's reply
    answer = response.choices[0].message.content

    return answer

if __name__ == "__main__":
  
    print("RAG System Active. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        # 1. Capture user input
        test_question = input("Ask a question about the document: ").strip()
        
        # 2. Check for exit condition
        if test_question.lower() in ['exit', 'quit']:
            print("Shutting down. Goodbye!")
            break
            
        # 3. Skip empty inputs to prevent accidental API crashes
        if not test_question:
            continue

        # 4. Process the query
        chunks, distances, search_queries = retrieve_relevant_chunks(test_question, top_k=4)

        print("\nExpanded search queries:")
        for i, search_query in enumerate(search_queries):
            print(f"  {i + 1}. {search_query}")
        print()

        print(f"\nTop {len(chunks)} retrieved chunks:\n")
        for i, (chunk_text, distance) in enumerate(zip(chunks, distances)):
            print(f"--- Chunk {i + 1} (distance: {distance:.4f}) ---")
            print(chunk_text)
            print()

        print("Generating answer...\n")
        answer = generate_answer(test_question, chunks)

        print("=== Answer ===")
        print(answer)
        print("-" * 40 + "\n")  # Visual separator for the next turn