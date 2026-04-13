import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    MAX_FILE_SIZE_BYTES,
    RATE_LIMIT_QUERY,
    RATE_LIMIT_UPLOAD,
    SUPPORTED_EXTENSIONS,
)
from ingestion import get_job, ingest_document, list_jobs
from llm import generate_answer
from models import (
    DocumentStatus,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from retrieval import retrieve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG QnA API starting up")
    yield
    logger.info("RAG QnA API shutting down")


app = FastAPI(
    title="RAG Question Answering API",
    description=(
        "Upload documents (PDF, TXT) and ask questions. "
        "Powered by Cohere embeddings, ChromaDB, and Groq LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["Documents"],
    summary="Upload a document for ingestion",
)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF or TXT file. Ingestion (chunking + embedding + storing)
    runs as a background job. Poll `/documents/{document_id}` to check status.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Read and validate size
    file_data = await file.read()
    if len(file_data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    if not file_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = str(uuid.uuid4())

    background_tasks.add_task(
        ingest_document,
        document_id=document_id,
        filename=file.filename,
        file_data=file_data,
        extension=suffix,
    )

    logger.info("Queued ingestion for doc_id=%s filename=%s", document_id, file.filename)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="queued",
        message="Document accepted. Ingestion running in background.",
    )


# ---------------------------------------------------------------------------
# Document status
# ---------------------------------------------------------------------------

@app.get(
    "/documents/{document_id}",
    response_model=DocumentStatus,
    responses={404: {"model": ErrorResponse}},
    tags=["Documents"],
    summary="Check ingestion status of a document",
)
async def document_status(document_id: str):
    job = get_job(document_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return DocumentStatus(**job)


@app.get(
    "/documents",
    response_model=list[DocumentStatus],
    tags=["Documents"],
    summary="List all ingested documents",
)
async def list_documents():
    return [DocumentStatus(**j) for j in list_jobs()]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Query"],
    summary="Ask a question against uploaded documents",
)
@limiter.limit(RATE_LIMIT_QUERY)
async def query(request: Request, body: QueryRequest):
    """
    Retrieve relevant chunks and generate an answer via Groq.

    Latency breakdown is included in the response for observability:
    - `embed_ms`: time to embed the query via Cohere
    - `search_ms`: time to run cosine similarity search in ChromaDB
    - `llm_ms`: time for Groq to generate the answer
    - `total_ms`: wall-clock end-to-end
    """
    import time
    t_total = time.perf_counter()

    # If scoped to a specific document, verify it's ready
    if body.document_id:
        job = get_job(body.document_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Document '{body.document_id}' not found.")
        if job["status"] != "ready":
            raise HTTPException(
                status_code=503,
                detail=f"Document is not ready yet (status: {job['status']}). Try again shortly.",
            )

    # Retrieve
    loop = asyncio.get_event_loop()
    chunks, retrieval_latency = await loop.run_in_executor(
        None, retrieve, body.question, body.top_k, body.document_id
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Ensure documents have been successfully ingested.",
        )

    # Generate
    answer, llm_ms = await loop.run_in_executor(
        None, generate_answer, body.question, chunks
    )

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)

    latency = {
        **retrieval_latency,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
    }

    logger.info(
        "Query answered | total=%.1fms embed=%.1fms search=%.1fms llm=%.1fms",
        total_ms,
        retrieval_latency["embed_ms"],
        retrieval_latency["search_ms"],
        llm_ms,
    )

    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=chunks,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
