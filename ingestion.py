import asyncio
import io
import logging
from datetime import datetime, timezone
from typing import Any

import cohere
import pdfplumber
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COHERE_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE_DOC,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients (module-level singletons)
# ---------------------------------------------------------------------------

_cohere_client: cohere.Client | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# In-memory job registry  {document_id: DocumentStatus-dict}
# ---------------------------------------------------------------------------

_job_registry: dict[str, dict[str, Any]] = {}


def get_job(document_id: str) -> dict[str, Any] | None:
    return _job_registry.get(document_id)


def list_jobs() -> list[dict[str, Any]]:
    return list(_job_registry.values())


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(data: bytes) -> str:
    """Extract all text from a PDF byte payload using pdfplumber."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


def extract_text_from_txt(data: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then latin-1."""
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("latin-1").strip()


def extract_text(data: bytes, extension: str) -> str:
    ext = extension.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(data)
    elif ext == ".txt":
        return extract_text_from_txt(data)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Strategy: token-aware sliding window using whitespace-split words as a proxy
# for tokens (roughly 1 word ≈ 1.3 tokens for English text).
# We use word count rather than a full tokeniser to avoid a heavyweight
# dependency; the approximation is sufficient for sentence-level coherence.
# See docs/design_decisions.md for full rationale.

_WORDS_PER_TOKEN_APPROX = 0.75  # 1 token ≈ 1/0.75 words


def _token_to_word_count(token_count: int) -> int:
    return max(1, int(token_count * _WORDS_PER_TOKEN_APPROX))


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping windows of ~CHUNK_SIZE tokens.
    Returns a list of non-empty chunk strings.
    """
    words = text.split()
    if not words:
        return []

    window = _token_to_word_count(CHUNK_SIZE)
    step = window - _token_to_word_count(CHUNK_OVERLAP)
    step = max(1, step)  # safety guard

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + window, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks using Cohere embed-v3.
    Uses input_type='search_document' so embeddings are optimised for storage
    and retrieval against query embeddings (which use 'search_query').
    """
    co = get_cohere()
    response = co.embed(
        texts=chunks,
        model=COHERE_EMBED_MODEL,
        input_type=COHERE_INPUT_TYPE_DOC,
    )
    return response.embeddings  # list[list[float]]


# ---------------------------------------------------------------------------
# ChromaDB write
# ---------------------------------------------------------------------------

def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    document_id: str,
    filename: str,
) -> int:
    """Write chunks + embeddings into ChromaDB. Returns number of chunks stored."""
    collection = get_collection()

    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Background ingestion job
# ---------------------------------------------------------------------------

async def ingest_document(
    document_id: str,
    filename: str,
    file_data: bytes,
    extension: str,
) -> None:
    """
    Full ingestion pipeline run in the background:
      1. Extract text
      2. Chunk
      3. Embed (via Cohere)
      4. Store in ChromaDB
    Updates _job_registry throughout.
    """
    _job_registry[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "status": "processing",
        "chunk_count": None,
        "ingested_at": None,
        "error": None,
    }

    try:
        # Run blocking I/O in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        logger.info("[%s] Extracting text from %s", document_id, filename)
        text = await loop.run_in_executor(None, extract_text, file_data, extension)

        if not text.strip():
            raise ValueError("No extractable text found in document.")

        logger.info("[%s] Chunking text (%d chars)", document_id, len(text))
        chunks = await loop.run_in_executor(None, chunk_text, text)

        if not chunks:
            raise ValueError("Chunking produced zero chunks.")

        logger.info("[%s] Embedding %d chunks via Cohere", document_id, len(chunks))
        embeddings = await loop.run_in_executor(None, embed_chunks, chunks)

        logger.info("[%s] Storing in ChromaDB", document_id)
        count = await loop.run_in_executor(
            None, store_chunks, chunks, embeddings, document_id, filename
        )

        _job_registry[document_id].update(
            {
                "status": "ready",
                "chunk_count": count,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("[%s] Ingestion complete — %d chunks stored", document_id, count)

    except Exception as exc:
        logger.exception("[%s] Ingestion failed: %s", document_id, exc)
        _job_registry[document_id].update(
            {"status": "failed", "error": str(exc)}
        )
