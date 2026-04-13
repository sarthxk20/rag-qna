import logging
import time
from typing import Optional

import cohere

from config import (
    COHERE_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE_QUERY,
    TOP_K,
)
from ingestion import get_collection
from models import RetrievedChunk

logger = logging.getLogger(__name__)

_cohere_client: cohere.Client | None = None


def get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def embed_query(question: str) -> tuple[list[float], float]:
    """
    Embed the user query using Cohere embed-v3 with input_type='search_query'.

    Cohere v3 uses separate embedding spaces for documents (search_document)
    and queries (search_query). Mixing them degrades retrieval quality — this
    asymmetric design is a core feature of the model, not a quirk.

    Returns (embedding, latency_ms).
    """
    co = get_cohere()
    t0 = time.perf_counter()
    response = co.embed(
        texts=[question],
        model=COHERE_EMBED_MODEL,
        input_type=COHERE_INPUT_TYPE_QUERY,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.embeddings[0], latency_ms


def similarity_search(
    query_embedding: list[float],
    top_k: int = TOP_K,
    document_id: Optional[str] = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    Query ChromaDB for the top-k most similar chunks.

    If document_id is provided, filters results to that document only
    (useful when the user has uploaded multiple documents and wants
    scoped answers).

    Returns (chunks, latency_ms).

    ChromaDB returns cosine distances in [0, 2]; we convert to similarity
    scores in [0, 1] via: similarity = 1 - (distance / 2).
    """
    collection = get_collection()

    where_filter = {"document_id": {"$eq": document_id}} if document_id else None

    t0 = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    chunks: list[RetrievedChunk] = []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        # Convert cosine distance → similarity score
        similarity = round(1.0 - dist / 2.0, 4)
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                text=text,
                similarity_score=similarity,
            )
        )

    logger.info(
        "Retrieved %d chunks (top score: %.4f) in %.1fms",
        len(chunks),
        chunks[0].similarity_score if chunks else 0.0,
        latency_ms,
    )

    return chunks, latency_ms


def retrieve(
    question: str,
    top_k: int = TOP_K,
    document_id: Optional[str] = None,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """
    Full retrieval step: embed query → similarity search.
    Returns (chunks, {embed_ms, search_ms}).
    """
    embedding, embed_ms = embed_query(question)
    chunks, search_ms = similarity_search(embedding, top_k=top_k, document_id=document_id)
    return chunks, {"embed_ms": round(embed_ms, 2), "search_ms": round(search_ms, 2)}
