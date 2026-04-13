from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str = "queued"
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: Optional[str] = Field(
        default=None,
        description="Scope retrieval to a specific document. If omitted, searches all documents.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: dict[str, float] = Field(
        description="Breakdown: embed_ms, search_ms, llm_ms, total_ms"
    )


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str          # queued | processing | ready | failed
    chunk_count: Optional[int] = None
    ingested_at: Optional[datetime] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
