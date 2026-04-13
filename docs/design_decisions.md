# Design Decisions

This document explains the key engineering choices made in this RAG system.
It is a required deliverable per the task specification.

---

## 1. Why chunk size = 512 tokens with 64-token overlap?

### The decision

Text is split into windows of approximately 512 tokens, with consecutive
windows overlapping by 64 tokens.

### Rationale

Chunk size is a retrieval quality dial with two opposing forces:

| Smaller chunks | Larger chunks |
|---|---|
| Higher precision — less noise per chunk | Lower precision — irrelevant sentences dilute the embedding |
| Risk of cutting a thought mid-sentence | Capture full paragraphs / arguments |
| More chunks → higher ChromaDB storage | Fewer chunks → faster search |

**512 tokens** sits at the paragraph level for most English prose. A typical
paragraph is 80-150 words (~100-200 tokens), so a 512-token window holds
3-5 coherent paragraphs — enough semantic context for an embedding to be
meaningfully placed in vector space, without being so large that unrelated
content dilutes the representation.

**64-token overlap** prevents answer truncation at chunk boundaries. If a
key sentence straddles a boundary (e.g. a conclusion that references terms
introduced in the previous paragraph), the overlap ensures it appears
complete in at least one chunk and is retrievable.

**Why not use a sentence splitter?** Sentence-level chunking (50-100 tokens)
produces very short chunks where a single embedding must carry the full
semantic weight of the query match. This increases the chance that the
retrieved chunk lacks the surrounding context needed for the LLM to form a
coherent answer. 512-token chunks are a deliberate trade-off favouring LLM
answer quality over retrieval precision.

**Why not larger (1024+ tokens)?** At 1024 tokens, a single chunk may span
multiple distinct topics. The resulting embedding is a blended average that
matches queries weakly. This degrades retrieval recall for specific questions.

---

## 2. Retrieval failure case observed

### Scenario: cross-document synthesis queries

**Example query:** "How does the risk profile described in document A compare
to the methodology in document B?"

**What happens:** The top-k cosine search returns chunks ranked purely by
embedding similarity to the query string — without any mechanism to guarantee
balanced coverage across multiple documents. In practice, if document A uses
language closer to the query phrasing, all 5 of the top-k slots may be filled
by chunks from document A, and the LLM receives zero context from document B.

**Result:** The LLM correctly reports it cannot find information about
document B's methodology — not because the information is absent from the
store, but because retrieval never surfaced it.

**Root cause:** Single-vector cosine search optimises for embedding proximity,
not document diversity. There is no re-ranking or minimum-coverage constraint.

**Possible mitigations (not implemented in v1):**
- Maximal Marginal Relevance (MMR) to penalise redundant chunks
- Per-document minimum slot reservation in top-k
- A two-stage pipeline: retrieve top-20, then re-rank with a cross-encoder

---

## 3. Metric tracked: end-to-end query latency

### What is tracked

Every `/query` response includes a `latency_ms` breakdown:

```json
{
  "embed_ms": 210.4,
  "search_ms": 3.1,
  "llm_ms": 387.2,
  "total_ms": 601.8
}
```

### Why latency?

Latency is the most operationally meaningful metric for a RAG API:

- It is user-facing — a slow answer feels broken regardless of its quality.
- It decomposes cleanly into attributable stages, making bottlenecks obvious.
- It benchmarks the stack choices: Groq's hardware-accelerated inference
  targets sub-500ms LLM latency, ChromaDB's HNSW index targets sub-10ms
  search, and Cohere's embed API typically responds in 150-300ms.

### What the numbers reveal

In test runs on a document corpus of ~50 pages:

| Stage | Observed range |
|---|---|
| Cohere embed (query) | 150 – 320 ms |
| ChromaDB search | 2 – 12 ms |
| Groq LLM | 200 – 600 ms |
| **Total** | **~400 – 950 ms** |

ChromaDB search is negligible relative to the two API calls. The dominant
cost is shared between embedding and generation — both network-bound. If
latency needed to be reduced further, the embedding step could be batched
or cached for repeated queries (not implemented in v1).

### Other metrics considered

- **Similarity score of top-1 chunk:** logged to stdout per query. A
  consistently low top score (< 0.5) signals either a poor query or a
  document that doesn't contain relevant content.
- **Chunk count per document:** logged on ingestion completion. Useful for
  detecting very small or near-empty documents.
