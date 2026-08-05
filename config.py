# config.py
# This file holds simple configuration values used across the project.
# Keeping config in one place means we don't hardcode paths/settings everywhere.

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and puts its values into environment variables
# This is how we keep secrets (like API keys) out of our source code
load_dotenv()

# The API key OpenAI gives us to authenticate our requests
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Path to the markdown document we want our chatbot to answer questions about
DOCUMENT_PATH = "data/document.md"

# Where ChromaDB will store its local database files
CHROMA_DB_PATH = "chroma_db"

# Name of the collection (like a "table") inside ChromaDB
COLLECTION_NAME = "simple_rag_collection"

# Name of the embedding model we'll use to turn text into vectors
EMBEDDING_MODEL = "text-embedding-3-small"

# Name of the chat model used to generate the final answer
CHAT_MODEL = "gpt-4.1-nano"