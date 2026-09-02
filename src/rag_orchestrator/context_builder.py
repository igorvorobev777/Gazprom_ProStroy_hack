from .schemas import RetrievedChunk


def assign_source_ids(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk.model_copy(update={"source_id": f"S{index}"})
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Собирает контекст, который видит LLM.

    Внутренние doc_id/chunk_id намеренно не передаются модели: для цитирования
    она должна использовать только короткие метки S1, S2 и т. д.
    """
    blocks: list[str] = []
    for chunk in chunks:
        lines = [
            f"SOURCE_ID: {chunk.source_id or 'UNKNOWN'}",
            f"[{chunk.source_id or 'UNKNOWN'}]",
            f"Документ: {chunk.title}",
        ]
        if chunk.section:
            lines.append(f"Раздел: {chunk.section}")
        lines.extend(["Текст:", chunk.text.strip()])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_source_map(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {chunk.source_id: chunk for chunk in chunks if chunk.source_id}


def build_compact_context(chunks: list[RetrievedChunk]) -> str:
    """Более короткий формат контекста для fast naive."""
    blocks: list[str] = []
    for chunk in chunks:
        # One-line blocks save a surprising number of prompt tokens on CPU while
        # preserving the title signal and citation id. Section is omitted here: it
        # is useful for UI/source metadata but rarely changes the answer itself.
        header = f"[{chunk.source_id or 'UNKNOWN'}] {chunk.title}"
        blocks.append(f"{header}: {chunk.text.strip()}")
    return "\n".join(blocks)
