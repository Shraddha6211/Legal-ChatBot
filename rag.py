# rag.py
# This script handles query-time logic: given a user's question,
# retrieve relevant chunks and (later) generate an answer.

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


def retrieve_relevant_chunks(question, top_k=15):
    """
    Given a user's question, returns the top_k most semantically
    similar chunks from the vector database.
    """
    # Step 1: embed the question using the same model as our chunks
    question_embedding = get_embedding(question)

    # Step 2: connect to the stored collection
    collection = get_collection()

    # Step 3: ask ChromaDB for the nearest neighbors to this question vector
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    # results is a dictionary with lists nested inside lists (one outer list
    # per query — we only sent one question, so we take index [0])
    retrieved_texts = results["documents"][0]
    distances = results["distances"][0]

    return retrieved_texts, distances

def build_prompt(question, retrieved_chunks):
    """
    Combines the retrieved chunks and the user's question into a single
    prompt string, with clear instructions for the model to stay grounded
    in the provided context.
    """
    # Join the chunks together with a separator so the model can tell
    # where one chunk ends and another begins
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- Only use information found in the "Context" section to answer.
- Do not use any outside knowledge, even if you know the answer.
- Answer the question using the provided context.
- If the answer can be reasonably inferred from the context, answer it.
- Only say "I couldn't find that information in the provided document." when the answer is clearly absent.

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
        chunks, distances = retrieve_relevant_chunks(test_question, top_k=15)

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