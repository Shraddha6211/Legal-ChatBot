# ingest.py
# This script is responsible for turning our raw document into something
# the chatbot can search through. Right now, it only does Step 1: reading the file.

import config
import chromadb
from openai import OpenAI 

# Create a client object that handles authenticated requests to OpenAI's API
client = OpenAI(api_key=config.OPENAI_API_KEY)


def read_markdown_file(file_path):
    """
    Opens a markdown file and returns its entire contents as a single string.

    Why a function? Even though this is one line of real logic, wrapping it
    in a function gives it a name, makes it reusable, and makes it easy to
    test or swap out later (e.g. if we read from a URL instead of disk).
    """
    # "r" means read mode (as opposed to "w" for write)
    # encoding="utf-8" makes sure special characters (accents, symbols) are read correctly
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()  # reads the ENTIRE file into one string

    return content

def split_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Splits a long string into a list of smaller overlapping chunks.

    chunk_size: how many characters each chunk should contain
    overlap: how many characters from the end of one chunk are repeated
             at the start of the next chunk

    Why overlap? It prevents ideas that fall on a chunk boundary from being
    completely severed between two chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # end is where this chunk stops
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the start forward, but step back by `overlap` characters
        # so the next chunk repeats a bit of this one
        start = end - overlap

    return chunks


def get_embedding(text):
    """
    Sends a piece of text to OpenAI's embedding model and returns
    the resulting vector (a list of floats).
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )

    # The API returns a list of embedding objects (one per input text).
    # Since we only sent one piece of text, we grab the first (and only) result.
    embedding_vector = response.data[0].embedding

    return embedding_vector
  
def build_vector_database(chunks):
    """
    Embeds every chunk and stores it in a local ChromaDB collection,
    persisted to disk at config.CHROMA_DB_PATH.
    """
    # PersistentClient saves data to disk (as opposed to an in-memory-only client)
    # so our database survives between separate runs of the program.
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # get_or_create_collection: if the collection already exists (from a previous
    # run), reuse it. Otherwise, create a new one.
    # We explicitly set the similarity metric to cosine, matching what we learned.
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(chunks)} chunks. This may take a minute...")

    for i, chunk_text in enumerate(chunks):
        embedding = get_embedding(chunk_text)

        # A unique string ID for this chunk, e.g. "chunk_0", "chunk_1", ...
        chunk_id = f"chunk_{i}"

        # collection.add() stores the id, the embedding vector, and the
        # original text together as one record.
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk_text]
        )

        # Simple progress indicator so we know it's not frozen
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    print("All chunks embedded and stored in ChromaDB.")
    return collection


if __name__ == "__main__":
    # This block only runs when you execute `python ingest.py` directly
    # (it won't run if this file is imported elsewhere)

    # document_text = read_markdown_file(config.DOCUMENT_PATH)

    # print(f"Successfully read document.")
    # print(f"Total characters: {len(document_text)}")

    # chunks = split_into_chunks(document_text, chunk_size=1000, overlap=200)

    # print(f"\nSplit into {len(chunks)} chunks.")
    # print("\n--- First chunk ---")
    # print(chunks[0])
    # print("\n--- Second chunk (notice the overlap at the start) ---")
    # print(chunks[1])

    # document_text = read_markdown_file(config.DOCUMENT_PATH)
    # print(f"Successfully read document.")
    # print(f"Total characters: {len(document_text)}")

    # chunks = split_into_chunks(document_text, chunk_size=1000, overlap=200)
    # print(f"\nSplit into {len(chunks)} chunks.")

    # # Test embedding on just the first chunk
    # print("\nGenerating embedding for the first chunk...")
    # test_embedding = get_embedding(chunks[0])

    # print(f"Embedding generated successfully.")
    # print(f"Vector length (number of dimensions): {len(test_embedding)}")
    # print(f"First 5 values: {test_embedding[:5]}")

    document_text = read_markdown_file(config.DOCUMENT_PATH)
    print(f"Successfully read document.")
    print(f"Total characters: {len(document_text)}")

    chunks = split_into_chunks(document_text, chunk_size=1000, overlap=200)
    print(f"Split into {len(chunks)} chunks.")

    collection = build_vector_database(chunks)

    print(f"\nCollection now contains {collection.count()} items.")