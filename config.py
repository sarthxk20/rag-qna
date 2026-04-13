import os
from dotenv import load_dotenv

load_dotenv()

# API keys
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Cohere
COHERE_EMBED_MODEL: str = "embed-english-v3.0"
COHERE_INPUT_TYPE_DOC: str = "search_document"
COHERE_INPUT_TYPE_QUERY: str = "search_query"

# ChromaDB
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME: str = "rag_documents"

# Groq
GROQ_MODEL: str = "llama3-70b-8192"
GROQ_MAX_TOKENS: int = 1024
GROQ_TEMPERATURE: float = 0.2

# Chunking
CHUNK_SIZE: int = 512       # tokens
CHUNK_OVERLAP: int = 64     # tokens

# Retrieval
TOP_K: int = 5              # chunks returned per query

# Rate limiting
RATE_LIMIT_UPLOAD: str = "5/minute"
RATE_LIMIT_QUERY: str = "10/minute"

# Supported file types
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt"}

# Max file size: 20 MB
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024
