import logging
import time

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)
from models import RetrievedChunk

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. You will be given:
1. A user question.
2. A set of relevant context passages retrieved from one or more uploaded documents.

Rules:
- Answer using ONLY the information present in the context passages.
- If the answer cannot be found in the context, say: "I could not find an answer to that question in the provided documents."
- Cite the source filename in your answer when referencing a specific fact, e.g. (source: report.pdf).
- Be concise. Do not pad your answer.
- Do not make up information.
"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Construct the user-turn message that packages retrieved chunks
    alongside the question.

    Format:
        Context passages:
        [1] (report.pdf, score=0.87)
        <text>

        [2] ...

        Question: <question>
    """
    context_blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{i}] (source: {chunk.filename}, similarity: {chunk.similarity_score:.4f})\n{chunk.text}"
        )

    context_section = "\n\n".join(context_blocks)
    return f"Context passages:\n\n{context_section}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, float]:
    """
    Call Groq's chat completion API with the built prompt.
    Returns (answer_text, latency_ms).

    Model choice: llama3-70b-8192
    - Groq's inference speed on this model averages ~200-400ms for typical
      RAG responses, making total query latency easily measurable and
      demonstrably fast. We use temperature=0.2 to keep answers grounded
      and reduce hallucination risk on factual retrieval tasks.
    """
    client = get_groq()
    user_message = build_prompt(question, chunks)

    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = completion.choices[0].message.content.strip()
    logger.info("LLM answered in %.1fms (%d output tokens)", latency_ms, completion.usage.completion_tokens)

    return answer, round(latency_ms, 2)
