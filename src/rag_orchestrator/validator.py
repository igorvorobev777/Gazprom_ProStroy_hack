from __future__ import annotations

import re

from .context_builder import build_context, build_source_map
from .llm_client import StructuredLLM
from .prompts import ANSWER_VALIDATOR_SYSTEM, answer_validator_prompt
from .schemas import (
    AnswerEvaluation,
    BasicValidation,
    GeneratedAnswer,
    RetrievedChunk,
    SourceReference,
)


SOURCE_PATTERN = re.compile(r"\[\s*[SsСс]\s*(\d+)\s*\]")
URL_PATTERN = re.compile(r"https?://\S+", flags=re.I)


def extract_source_ids(answer: str) -> list[str]:
    result: list[str] = []
    for number in SOURCE_PATTERN.findall(answer):
        source_id = f"S{number}"
        if source_id not in result:
            result.append(source_id)
    return result


def validate_answer_format(
    generated: GeneratedAnswer,
    chunks: list[RetrievedChunk],
) -> BasicValidation:
    issues: list[str] = []
    answer = generated.answer.strip()
    allowed = {chunk.source_id for chunk in chunks if chunk.source_id}
    cited = set(extract_source_ids(answer))
    declared = set(generated.used_source_ids)

    if not answer:
        issues.append("Ответ пустой.")
    if cited - allowed:
        issues.append(f"Несуществующие ссылки в тексте: {sorted(cited - allowed)}")
    if declared - allowed:
        issues.append(f"Несуществующие used_source_ids: {sorted(declared - allowed)}")
    if not generated.insufficient_context and not cited:
        issues.append("Ответ не содержит ссылок [S1], [S2] и т.д.")
    if cited != declared:
        issues.append("Ссылки в answer не совпадают с used_source_ids.")
    if URL_PATTERN.search(answer):
        issues.append("Модель вставила URL самостоятельно.")

    return BasicValidation(valid=not issues, issues=issues)


def collect_semantic_issues(evaluation: AnswerEvaluation) -> list[str]:
    issues: list[str] = []

    if not evaluation.valid:
        issues.append(
            evaluation.reason.strip()
            or "Семантический валидатор отклонил ответ без объяснения."
        )
    if not evaluation.answers_question:
        issues.append("Ответ не раскрывает все части вопроса.")
    if not evaluation.grounded_in_context:
        issues.append("Ответ не полностью подтверждён переданными источниками.")
    issues.extend(item.strip() for item in evaluation.unsupported_claims if item.strip())
    issues.extend(item.strip() for item in evaluation.missing_citations if item.strip())

    # Даже противоречивое valid=true не должно скрывать остальные поля.
    return list(dict.fromkeys(issues))


def validate_answer_with_llm(
    query: str,
    chunks: list[RetrievedChunk],
    answer: str,
    llm: StructuredLLM,
) -> AnswerEvaluation:
    return llm.structured(
        response_model=AnswerEvaluation,
        system_prompt=ANSWER_VALIDATOR_SYSTEM,
        user_prompt=answer_validator_prompt(query, build_context(chunks), answer),
        max_tokens=260,
    )


def resolve_sources(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[SourceReference]:
    source_map = build_source_map(chunks)
    ordered_ids = extract_source_ids(answer)

    result: list[SourceReference] = []
    for source_id in ordered_ids:
        chunk = source_map.get(source_id)
        if not chunk:
            continue
        result.append(
            SourceReference(
                source_id=source_id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
                url=chunk.url,
                score=chunk.score,
            )
        )
    return result
