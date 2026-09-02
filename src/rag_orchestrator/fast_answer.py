from __future__ import annotations

import re

from .context_builder import build_compact_context
from .fast_context import roots
from .schemas import GeneratedAnswer, RetrievedChunk

FAST_ANSWER_SYSTEM = """
Отвечай по-русски только по SOURCES. Без повтора вопроса и вводных фраз.
Для обычных вопросов давай развернутый ответ из 5-10 предложений. Если в источнике есть несколько фактов, условий, правил, этапов или исключений — обязательно раскрой их. Для процедур используй структурированные пункты. Не сокращай ответ до одной фразы.
Процедура: опиши последовательность действий, роли, условия и исключения, если они есть.
Сохраняй термины, числа, роли и условия источника; внешние знания запрещены.
Служебные [ОРГАНИЗАЦИЯ_N], [ЧЕЛОВЕК_N], [АДРЕС_N] опускай, если смысл сохраняется.
После каждого фактического предложения ставь одну реальную ссылку вида [S1] или [S2]. Никогда не печатай буквальный текст S#.
Если объект вопроса неясен при разных источниках — попроси уточнить объект.
Не отвечай одним предложением, если вопрос требует объяснения. Добавляй контекст, условия и детали из источника.
Заверши последнее предложение до достижения лимита токенов.
/no_think
""".strip()

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
_SOURCE_MARKER = re.compile(r"\[\s*[SsСс]\s*(\d+)\s*\]")


def build_fast_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_compact_context(chunks)
    return f"Q: {query}\nSOURCES:\n{context}\nA:"


def _strip_question_echo(text: str, query: str) -> str:
    if not query:
        return text.strip()
    clean_query = re.sub(r"\s+", " ", query).strip(" \t\r\n")
    candidate = text.lstrip()
    # Qwen sometimes echoes the user question verbatim before answering.
    if candidate.casefold().startswith(clean_query.casefold()):
        candidate = candidate[len(clean_query):].lstrip(" \t\r\n:-—")
    return candidate.strip()


def _trim_incomplete_tail(text: str) -> str:
    """Keep only complete cited sentences when llama.cpp reports finish_reason=length."""
    if not text:
        return text

    # Drop a tail cut inside a bracket/placeholder first.
    balance = 0
    last_balanced = 0
    for idx, ch in enumerate(text):
        if ch == "[":
            balance += 1
        elif ch == "]" and balance > 0:
            balance -= 1
        if balance == 0:
            last_balanced = idx + 1
    if balance:
        text = text[:last_balanced].rstrip()

    # A citation alone is not proof that the sentence is complete: Qwen can emit
    # "... по телефон [S1]" exactly at max_tokens. Keep a citation only when its
    # statement has terminal punctuation immediately before or after the marker.
    safe_end = 0
    for match in re.finditer(r"\[S\d+\]", text):
        before = text[:match.start()].rstrip()
        after = text[match.end():].lstrip()
        end = match.end()
        complete = bool(before and before[-1] in ".!?")
        if after and after[0] in ".!?":
            complete = True
            end = match.end() + (len(text[match.end():]) - len(after)) + 1
        if complete and text[:end].count("[") == text[:end].count("]"):
            safe_end = end
    if safe_end:
        return text[:safe_end].rstrip()

    # No complete cited sentence survived; returning empty forces the safer
    # extractive fallback instead of exposing a syntactically broken statement.
    return ""


def _posthoc_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]] | None:
    """Attach citations only when a sentence can be locally aligned to evidence.

    This is safer than blindly appending [S1] to an uncited model answer. It is
    used only when Qwen forgot all citation markers; no extra LLM call is needed.
    """
    chunk_features = [
        (chunk.source_id or "", roots(f"{chunk.title} {chunk.text}"), rank)
        for rank, chunk in enumerate(chunks)
        if chunk.source_id
    ]
    if not chunk_features:
        return None

    parts = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    if not parts:
        return None

    cited_parts: list[str] = []
    used: list[str] = []
    aligned = 0
    for sentence in parts:
        if _SOURCE_MARKER.search(sentence):
            cited_parts.append(sentence)
            continue
        sentence_roots = roots(sentence)
        if not sentence_roots:
            cited_parts.append(sentence)
            continue

        best_id = ""
        best_score = 0.0
        for source_id, source_roots, rank in chunk_features:
            overlap = len(sentence_roots & source_roots)
            if not overlap:
                continue
            coverage = overlap / len(sentence_roots)
            # Prefer stronger retrieval sources when lexical support is similar.
            score = coverage + 0.04 / (rank + 1)
            if score > best_score:
                best_score = score
                best_id = source_id

        # A low-overlap sentence may be an unsupported paraphrase. Do not create
        # a misleading citation; let the pipeline use the grounded fallback.
        if not best_id or best_score < 0.28:
            return None
        cited_parts.append(f"{sentence.rstrip()} [{best_id}]")
        aligned += 1
        if best_id not in used:
            used.append(best_id)

    if not aligned:
        return None
    return " ".join(cited_parts), used


def normalize_plain_answer(
    text: str,
    chunks: list[RetrievedChunk],
    *,
    query: str = "",
    truncated: bool = False,
    allow_posthoc_citations: bool = True,
) -> GeneratedAnswer:
    text = _strip_question_echo(text.strip(), query)
    # Small Qwen models sometimes literalize the documentation placeholder "S#".
    text = re.sub(r"^\s*(?:S#|Ответ\s*:|A\s*:)\s*", "", text, flags=re.I).strip()
    text = re.sub(r"(?<!\[)\bS#\s*", "", text, flags=re.I)
    if truncated:
        text = _trim_incomplete_tail(text)
    # Remove empty citation-only list labels such as "1. [S1]".
    text = re.sub(r"(?m)(?:^|\s)\d+[.)]\s*\[S\d+\](?=\s|$)", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    allowed = {chunk.source_id for chunk in chunks if chunk.source_id}
    ids: list[str] = []
    for number in _SOURCE_MARKER.findall(text):
        source_id = f"S{number}"
        if source_id in allowed and source_id not in ids:
            ids.append(source_id)
    text = _SOURCE_MARKER.sub(lambda m: f"[S{m.group(1)}]", text)

    if text and not ids and allow_posthoc_citations:
        aligned = _posthoc_citations(text, chunks)
        if aligned is not None:
            text, ids = aligned

    if not text or not ids:
        return GeneratedAnswer(
            answer="В переданных источниках недостаточно информации для надёжного ответа.",
            used_source_ids=[],
            insufficient_context=True,
        )

    return GeneratedAnswer(
        answer=text,
        used_source_ids=ids,
        insufficient_context=False,
    )
