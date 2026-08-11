# ingest.py
# This script is responsible for turning our raw document into something
# the chatbot can search through. Right now, it only does Step 1: reading the file.

import config
import chromadb
from openai import OpenAI 
from chunking import build_chunks 

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
    UPDATED: chunks is now a list of dicts (text + metadata) from build_chunks(),
    instead of a plain list of strings. We now also store metadata in ChromaDB.
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

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        # collection.add() stores the id, the embedding vector, and the
        # original text together as one record.
        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )

        # Simple progress indicator so we know it's not frozen
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    print("All chunks embedded and stored in ChromaDB.")
    return collection


if __name__ == "__main__":
    # This block only runs when you execute `python ingest.py` directly
    # (it won't run if this file is imported elsewhere)
    document_text = read_markdown_file(config.DOCUMENT_PATH)
    print(f"Successfully read document.")
    print(f"Total characters: {len(document_text)}")

    # CHANGED: use the new recursive Markdown chunker instead of split_into_chunks
    chunks = build_chunks(
        document_text,
        source_name=config.DOCUMENT_PATH,
        max_tokens=1500,
        overlap_tokens=300
    )
    print(f"Split into {len(chunks)} chunks.")

    # Quick sanity check: print the first chunk's metadata
    print("\nExample chunk metadata:")
    print(chunks[0]["metadata"])
    print("\nExample chunk text (first 200 chars):")
    print(chunks[0]["text"][:200])

    collection = build_vector_database(chunks)

    print(f"\nCollection now contains {collection.count()} items.")
    

   