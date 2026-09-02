from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from rag_orchestrator.config import Settings


class Embedder(Protocol):
    dimension: int
    backend_name: str
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Быстрый CPU backend без скачивания моделей.

    Это быстрый локальный режим. Для настоящей семантики включается
    sentence_transformers через EMBEDDING_BACKEND.
    """

    backend_name = "hashing"
    model_name = "word+char-hashing"

    def __init__(self, dimension: int = 768) -> None:
        if dimension % 2:
            dimension += 1
        self.dimension = dimension
        half = dimension // 2
        self._word = HashingVectorizer(
            n_features=half,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self._char = HashingVectorizer(
            n_features=half,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        word = self._word.transform(texts)
        char = self._char.transform(texts)
        matrix = sparse.hstack([word, char], format="csr")
        matrix = normalize(matrix, norm="l2", axis=1)
        return matrix.toarray().astype(np.float32, copy=False)


class SentenceTransformerEmbedder:
    backend_name = "sentence_transformers"

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Для EMBEDDING_BACKEND=sentence_transformers установи "
                "requirements-models.txt"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            settings.embedding_model,
            settings.embedding_device,
        )
    return HashingEmbedder(settings.embedding_dimension)
