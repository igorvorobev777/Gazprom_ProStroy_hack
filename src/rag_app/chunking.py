from __future__ import annotations

import re

from .models import ChunkRecord, KnowledgeDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def _normalize_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _word_safe_slices(text: str, max_chars: int) -> list[str]:
    """Split an oversized unit without cutting words whenever possible."""
    result: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        cut = rest.rfind(" ", max(1, int(max_chars * 0.65)), max_chars + 1)
        if cut < 0:
            cut = max_chars
        result.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        result.append(rest)
    return result


def _semantic_units(text: str, max_chars: int) -> list[str]:
    """Preserve HTML-derived block boundaries and sentence boundaries.

    HiHub HTML conversion keeps block tags as newlines. The previous chunker
    flattened those newlines and created overlap by arbitrary characters, which
    produced fragments that started in the middle of words. These units let us
    overlap whole semantic pieces instead.
    """
    units: list[str] = []
    blocks = [_normalize_line(block) for block in re.split(r"\n+", text) if _normalize_line(block)]
    for block in blocks:
        if len(block) <= max_chars:
            units.append(block)
            continue
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(block) if part.strip()]
        if len(sentences) <= 1:
            units.extend(_word_safe_slices(block, max_chars))
            continue
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    units.append(" ".join(current))
                    current, current_len = [], 0
                units.extend(_word_safe_slices(sentence, max_chars))
                continue
            extra = len(sentence) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                units.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += extra
        if current:
            units.append(" ".join(current))
    return units


def _pack_units(units: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        extra = len(unit) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(current)
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += extra
    if current:
        chunks.append(current)
    return chunks


def _tail_units(units: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    selected: list[str] = []
    size = 0
    for unit in reversed(units):
        extra = len(unit) + (1 if selected else 0)
        # Keep at least one whole unit if it is reasonably sized. Never slice it.
        if selected and size + extra > overlap_chars:
            break
        if not selected and len(unit) > overlap_chars * 2:
            break
        selected.append(unit)
        size += extra
        if size >= overlap_chars:
            break
    return list(reversed(selected))


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[ChunkRecord]:
    units = _semantic_units(document.content, chunk_size_chars)
    base_chunks = _pack_units(units, chunk_size_chars)
    if not base_chunks:
        return []

    records: list[ChunkRecord] = []
    previous_tail: list[str] = []
    for index, base_units in enumerate(base_chunks, start=1):
        combined = [*previous_tail, *base_units]
        text = " ".join(combined).strip()
        previous_tail = _tail_units(base_units, overlap_chars)

        records.append(
            ChunkRecord(
                doc_id=document.id,
                chunk_id=f"{document.id}:v{document.version}:chunk-{index}",
                version=document.version,
                title=document.title,
                section=document.section,
                text=text,
                url=document.url,
                updated_at=document.updated_at,
                metadata={
                    **document.metadata,
                    "chunk_number": index,
                    "is_actual": document.is_actual,
                    "semantic_boundaries": True,
                },
            )
        )
    return records


def chunk_documents(
    documents: list[KnowledgeDocument],
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[ChunkRecord]:
    result: list[ChunkRecord] = []
    for document in documents:
        result.extend(
            chunk_document(
                document,
                chunk_size_chars=chunk_size_chars,
                overlap_chars=overlap_chars,
            )
        )
    return result
