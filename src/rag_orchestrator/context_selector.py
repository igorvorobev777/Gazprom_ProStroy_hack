from __future__ import annotations

from .schemas import RetrievedChunk


def select_naive_context(
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Выбирает до top_k уникальных чанков без потери текста.

    Порядок reranker сохраняется.
    Дубли удаляются по паре doc_id/chunk_id.
    Текст, идентификаторы источников и metadata не меняются.
    """

    selected: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        if len(selected) >= top_k:
            break

        key = (
            chunk.doc_id,
            chunk.chunk_id,
        )

        if key in seen:
            continue

        seen.add(key)

        if not chunk.text.strip():
            continue

        selected.append(
            chunk.model_copy(deep=True)
        )

    return selected