from __future__ import annotations

import hashlib
import json
import shutil
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from rag_orchestrator.config import Settings

from .chunking import chunk_documents
from .embeddings import Embedder, build_embedder
from .knowledge_base import KnowledgeBaseSource
from .models import ChunkRecord, IndexManifest


MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.json"
DENSE_FILE = "dense_vectors.npy"
SPARSE_FILE = "sparse_index.joblib"
CHUNKING_VERSION = "semantic-boundaries-v3"
SPARSE_LIBRARY_VERSION = sklearn.__version__


def _corpus_hash(chunks: list[ChunkRecord]) -> str:
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "updated_at": chunk.updated_at.isoformat(),
        }
        for chunk in chunks
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_sparse_index(texts: list[str]):
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    token_pattern=r"(?u)\b\w\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                    max_features=25000,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 0.45},
    )
    matrix = vectorizer.fit_transform(texts)
    matrix = normalize(matrix, norm="l2", axis=1)
    return vectorizer, sparse.csr_matrix(matrix)


def build_index(
    *,
    source: KnowledgeBaseSource,
    settings: Settings,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IndexManifest:
    index_dir = settings.index_dir
    if index_dir.exists() and not force and (index_dir / MANIFEST_FILE).exists():
        return load_manifest(index_dir)

    documents = source.list_documents()
    chunks = chunk_documents(
        documents,
        chunk_size_chars=settings.chunk_size_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    if not chunks:
        raise ValueError("После chunking не осталось ни одного фрагмента.")

    embedder = embedder or build_embedder(settings)
    texts = [chunk.text for chunk in chunks]
    dense_vectors = embedder.encode(texts)
    if dense_vectors.shape != (len(chunks), embedder.dimension):
        raise ValueError("Embedding backend вернул массив неожиданной формы.")

    sparse_vectorizer, sparse_matrix = _build_sparse_index(texts)

    manifest = IndexManifest(
        source_key=source.source_key,
        chunking_version=CHUNKING_VERSION,
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        created_at=datetime.now(timezone.utc),
        document_count=len(documents),
        chunk_count=len(chunks),
        embedding_backend=embedder.backend_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        sparse_library="scikit-learn",
        sparse_library_version=SPARSE_LIBRARY_VERSION,
        corpus_hash=_corpus_hash(chunks),
    )

    # Build every artifact in a sibling temporary directory first.  The active
    # The RAG container may still be serving queries while a live sync triggers
    # SyncKnowledgeBase, so deleting the live directory before a successful
    # build would make a failed sync destructive.  A complete directory swap
    # keeps the old index restart-safe until the new index is fully written.
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = index_dir.parent / f".{index_dir.name}.build-{uuid4().hex}"
    backup_dir = index_dir.parent / f".{index_dir.name}.backup-{uuid4().hex}"
    build_dir.mkdir(parents=False, exist_ok=False)

    try:
        (build_dir / CHUNKS_FILE).write_text(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(build_dir / DENSE_FILE, dense_vectors.astype(np.float32))
        joblib.dump(
            {"vectorizer": sparse_vectorizer, "matrix": sparse_matrix},
            build_dir / SPARSE_FILE,
            compress=3,
        )
        (build_dir / MANIFEST_FILE).write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        had_previous = index_dir.exists()
        if had_previous:
            index_dir.rename(backup_dir)
        try:
            build_dir.rename(index_dir)
        except Exception:
            # If the second rename fails, restore the previously healthy index.
            if had_previous and backup_dir.exists() and not index_dir.exists():
                backup_dir.rename(index_dir)
            raise

        # The live LocalIndex keeps all arrays in process memory (no mmap), so
        # Windows can safely remove the old directory after the swap.  Cleanup
        # failure is non-fatal: the active canonical index is already valid.
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        return manifest
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


def load_manifest(index_dir: Path) -> IndexManifest:
    path = index_dir / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Индекс не готов: отсутствует {path}")
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


class LocalIndex:
    """Постоянный локальный vector+sparse index для RAG retrieval."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.manifest = load_manifest(index_dir)
        self.chunks = [
            ChunkRecord.model_validate(item)
            for item in json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        # Keep the relatively small dense matrix in RAM instead of memory-mapping it.
        # This is important on Windows: a mapped ``dense_vectors.npy`` can keep
        # the old index directory locked while a live reindex is running.
        self.dense_vectors = np.load(index_dir / DENSE_FILE)
        sparse_data = joblib.load(index_dir / SPARSE_FILE)
        self.sparse_vectorizer = sparse_data["vectorizer"]
        self.sparse_matrix = sparse.csr_matrix(sparse_data["matrix"])

        if self.dense_vectors.shape[0] != len(self.chunks):
            raise ValueError("Число dense-векторов не совпадает с числом чанков.")
        if self.sparse_matrix.shape[0] != len(self.chunks):
            raise ValueError("Число sparse-векторов не совпадает с числом чанков.")
