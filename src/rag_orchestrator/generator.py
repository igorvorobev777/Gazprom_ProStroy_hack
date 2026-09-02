from __future__ import annotations

import re

from .context_builder import build_context
from .llm_client import StructuredLLM
from .prompts import ANSWER_GENERATOR_SYSTEM, answer_generator_prompt
from .schemas import GeneratedAnswer, RetrievedChunk


FLEXIBLE_SOURCE_PATTERN = re.compile(r"\[\s*[SsСс]\s*(\d+)\s*\]")
CANONICAL_SOURCE_PATTERN = re.compile(r"\[(S\d+)\]")


def _canonicalize_bracket_sources(answer: str) -> str:
    return FLEXIBLE_SOURCE_PATTERN.sub(
        lambda match: f"[S{match.group(1)}]",
        answer,
    )


def normalize_generated_sources(
    result: GeneratedAnswer,
    chunks: list[RetrievedChunk],
) -> GeneratedAnswer:
    aliases: dict[str, str] = {}
    allowed_order: list[str] = []
    for chunk in chunks:
        if not chunk.source_id:
            continue
        allowed_order.append(chunk.source_id)
        for alias in (chunk.source_id, chunk.doc_id, chunk.chunk_id):
            aliases[alias.casefold()] = chunk.source_id

    answer = _canonicalize_bracket_sources(result.answer.strip())

    for alias_key, source_id in sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias_key == source_id.casefold():
            continue
        answer = re.sub(
            rf"\[\s*{re.escape(alias_key)}\s*\]",
            f"[{source_id}]",
            answer,
            flags=re.IGNORECASE,
        )

    cited_ids: list[str] = []
    for source_id in CANONICAL_SOURCE_PATTERN.findall(answer):
        if source_id not in cited_ids:
            cited_ids.append(source_id)

    declared_ids = list(dict.fromkeys(result.used_source_ids))

    if cited_ids:
        final_ids = [
            source_id for source_id in cited_ids if source_id in allowed_order
        ]
    elif declared_ids and not result.insufficient_context:
        citations = ", ".join(f"[{source_id}]" for source_id in declared_ids)
        answer = f"{answer.rstrip()}\n\nИсточники: {citations}".strip()
        final_ids = declared_ids
    else:
        final_ids = []

    return result.model_copy(
        update={
            "answer": answer,
            "used_source_ids": final_ids,
        }
    )


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: StructuredLLM,
    feedback: list[str] | None = None,
) -> GeneratedAnswer:
    available_source_ids = [
        chunk.source_id for chunk in chunks if chunk.source_id
    ]
    result = llm.structured(
        response_model=GeneratedAnswer,
        system_prompt=ANSWER_GENERATOR_SYSTEM,
        user_prompt=answer_generator_prompt(
            query,
            build_context(chunks),
            available_source_ids,
            feedback,
        ),
        max_tokens=350,
    )
    return normalize_generated_sources(result, chunks)
