from __future__ import annotations

from .context_builder import build_compact_context
from .generator import normalize_generated_sources
from .llm_client import StructuredLLM
from .prompts import FAST_NAIVE_SYSTEM, fast_naive_prompt
from .schemas import (
    FastNaiveDecision,
    FastNaiveResponse,
    GeneratedAnswer,
    RetrievedChunk,
)


def run_fast_naive(
    query: str,
    chunks: list[RetrievedChunk],
    llm: StructuredLLM,
    *,
    max_tokens: int,
) -> FastNaiveResponse:
    available_source_ids = [
        chunk.source_id for chunk in chunks if chunk.source_id
    ]

    result = llm.structured(
        response_model=FastNaiveResponse,
        system_prompt=FAST_NAIVE_SYSTEM,
        user_prompt=fast_naive_prompt(
            query,
            build_compact_context(chunks),
            available_source_ids,
        ),
        max_tokens=max_tokens,
    )

    unknown = set(result.used_source_ids) - set(available_source_ids)
    if unknown:
        raise ValueError(
            f"Fast naive указал несуществующие источники: {sorted(unknown)}"
        )

    if result.decision == FastNaiveDecision.ANSWER:
        normalized = normalize_generated_sources(
            GeneratedAnswer(
                answer=result.answer,
                used_source_ids=result.used_source_ids,
                insufficient_context=False,
            ),
            chunks,
        )
        result = result.model_copy(
            update={
                "answer": normalized.answer,
                "used_source_ids": normalized.used_source_ids,
            }
        )

    return result
