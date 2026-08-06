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

# def split_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Splits a long string into a list of smaller overlapping chunks.

    chunk_size: how many characters each chunk should contain
    overlap: how many characters from the end of one chunk are repeated
             at the start of the next chunk

    Why overlap? It prevents ideas that fall on a chunk boundary from being
    completely severed between two chunks.
    """

     # 1. Edge case handling: Return empty list if text is empty
    if not text or not text.strip():
        return []

    # 2. Initialize the smart recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""] # Priority: Para -> Line -> Sentence -> Word
    )
    # chunks = []
    # start = 0
    # text_length = len(text)

    # while start < text_length:
    #     # end is where this chunk stops
    #     end = start + chunk_size
    #     chunk = text[start:end]
    #     chunks.append(chunk)

    #     # Move the start forward, but step back by `overlap` characters
    #     # so the next chunk repeats a bit of this one
    #     start = end - overlap

    # 3. Split and return the text list directly
    chunks = splitter.split_text(text)

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
        max_tokens=2000,
        overlap_tokens=400
    )
    print(f"Split into {len(chunks)} chunks.")

    # Quick sanity check: print the first chunk's metadata
    print("\nExample chunk metadata:")
    print(chunks[0]["metadata"])
    print("\nExample chunk text (first 200 chars):")
    print(chunks[0]["text"][:200])

    collection = build_vector_database(chunks)

    print(f"\nCollection now contains {collection.count()} items.")
    # document_text = read_markdown_file(config.DOCUMENT_PATH)

    # print(f"Successfully read document.")
    # print(f"Total characters: {len(document_text)}")

    # chunks = build_chunks(
    #     document_text, 
    #     source_name=config.DOCUMENT_PATH,
    #     max_tokens=2000,
    #     overlap_tokens=400
    # )

    # print(f"Total chunks generated: {len(chunks)}")

    # # 2. Inspect multiple chunks to verify quality, overlap, and context
    # print("\n=== CHUNKING QUALITY CHECK ===")
    
    # # Check the first chunk, a middle chunk, and the last chunk
    # sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
    # # Handle small documents with fewer than 3 chunks safely
    # sample_indices = sorted(list(set(idx for idx in sample_indices if idx < len(chunks))))

    # for idx in sample_indices:
    #     print(f"\n" + "="*50)
    #     print(f"--- INSPECTING CHUNK INDEX: {idx} ---")
    #     print(f"="*50)
    #     print(f"Metadata: {chunks[idx]['metadata']}")
    #     print(f"Character Length: {len(chunks[idx]['text'])}")
    #     print(f"--- Chunk Content Start ---")
    #     print(chunks[idx]["text"])
    #     print(f"--- Chunk Content End ---")

   