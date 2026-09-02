from typing import Protocol

from .schemas import RetrievedChunk


class Retriever(Protocol):
    """Контракт retrieval-модуля для поиска релевантных чанков."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_id: int = 0,
    ) -> list[RetrievedChunk]: ...

    def search_multiple(
        self,
        queries: list[str],
        top_k: int = 5,
        section_id: int = 0,
    ) -> list[RetrievedChunk]: ...
