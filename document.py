import io
import re
import uuid

import chromadb
from openai import OpenAI
from pypdf import PdfReader

import config
from chunking import build_chunks

client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_collection():
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def normalize_pdf_text(raw_text: str) -> str:
    """Convert extracted PDF text into cleaner Markdown-style paragraphs."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Turn single line breaks into spaces while preserving paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_pdf_to_markdown(pdf_bytes: bytes, filename: str) -> str:
    """Extract text from a PDF and return a lightweight Markdown string."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = normalize_pdf_text(raw_text)
        if not clean_text:
            continue

        # Use a top-level page heading so the chunker can detect document structure.
        page_heading = f"# {filename} — Page {page_number}"
        pages.append(f"{page_heading}\n\n{clean_text}")

    return "\n\n".join(pages)


def generate_document_id(filename: str) -> str:
    return f"doc_{uuid.uuid4().hex}"


def build_document_chunks(markdown_text: str, filename: str, document_id: str):
    source_name = f"uploaded:{filename}:{document_id}"
    chunks = build_chunks(
        markdown_text,
        source_name=source_name,
        max_tokens=1500,
        overlap_tokens=300,
    )

    for chunk in chunks:
        chunk_id = f"{document_id}_{chunk['chunk_id']}"
        chunk["chunk_id"] = chunk_id
        chunk["metadata"]["document_id"] = document_id
        chunk["metadata"]["filename"] = filename
        chunk["metadata"]["source"] = source_name

    return chunks


def ingest_uploaded_document(pdf_bytes: bytes, filename: str):
    markdown_text = convert_pdf_to_markdown(pdf_bytes, filename)
    document_id = generate_document_id(filename)
    chunks = build_document_chunks(markdown_text, filename, document_id)
    collection = get_collection()

    print(f"Ingesting uploaded document '{filename}' with ID {document_id}...")

    for chunk in chunks:
        embedding = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=chunk["text"],
        )
        vector = embedding.data[0].embedding

        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "markdown_text": markdown_text,
    }


def delete_document(document_id: str):
    collection = get_collection()
    collection.delete(where={"document_id": document_id})


def get_document_chunks(document_id: str):
    collection = get_collection()
    results = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    documents = results.get("documents", [])
    if documents and isinstance(documents[0], list):
        documents = documents[0]
    metadatas = results.get("metadatas", [])
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]

    return documents, metadatas
