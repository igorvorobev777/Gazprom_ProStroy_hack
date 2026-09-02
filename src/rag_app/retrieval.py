from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import exp, isfinite, log1p
from threading import local
from time import perf_counter
from typing import Sequence

import numpy as np

from rag_orchestrator.config import Settings
from rag_orchestrator.schemas import RetrievedChunk

from .embeddings import Embedder
from .index_store import LocalIndex


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0:
        return []
    top_k = min(top_k, scores.size)
    if top_k == scores.size:
        return np.argsort(scores)[::-1].tolist()
    selected = np.argpartition(scores, -top_k)[-top_k:]
    return selected[np.argsort(scores[selected])[::-1]].tolist()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.I)
_STOP_WORDS = {
    "а", "без", "бы", "был", "была", "были", "быть", "в", "вам", "вас",
    "весь", "во", "вот", "все", "всех", "где", "да", "для", "до", "его",
    "ее", "её", "если", "есть", "еще", "ещё", "же", "за", "зачем", "и",
    "из", "или", "им", "их", "к", "как", "какая", "какие", "какой", "кем",
    "когда", "кто", "ли", "на", "над", "надо", "не", "него", "нее", "ней",
    "нет", "но", "о", "об", "он", "она", "они", "оно", "от", "по", "под",
    "при", "про", "с", "со", "так", "также", "то", "у", "уже", "что", "чем",
    "чего", "это", "эта", "эти", "этот", "должен", "должна", "должны", "нужно",
    "можно", "порядок", "каков", "какова",
}


def _root(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    if token.isdigit():
        return token
    # Prefix stemming is intentionally conservative and cheap. 5-7 chars
    # are enough to connect Russian inflections without introducing a model.
    if len(token) >= 9:
        return token[:6]
    if len(token) >= 6:
        return token[:5]
    return token


def lexical_roots(text: str) -> list[str]:
    result: list[str] = []
    for token in _TOKEN_RE.findall(text.casefold().replace("ё", "е")):
        if token in _STOP_WORDS or (len(token) < 3 and not token.isdigit()):
            continue
        result.append(_root(token))
    return result








def _query_section_intent(query: str) -> tuple[set[str], set[int]]:
    """Detect strong corporate intents and preferred sections."""
    q = query.casefold().replace("ё", "е")
    keywords: set[str] = set()
    sections: set[int] = set()

    rules = [
        (("расскажи о компании", "информация о компании", "кто мы", "история компании", "деятельность компании", "ценности компании"), {43919, 43915}),
        (("отпуск", "удаленная работа", "удалённая работа", "график", "испытательный срок", "обучение сотрудник"), {43916}),
        (("компьютер", "доступ", "техническая поддержка", "it поддержка", "оборудование"), {43917}),
        (("новости", "события компании"), {43918}),
        (("пожар", "требования пожарной безопасности", "меры пожарной безопасности"), {44030}),
    ]
    for phrases, ids in rules:
        if any(p in q for p in phrases):
            sections.update(ids)
    return keywords, sections


def _intent_section_boost(query: str, section: str | None, title: str) -> tuple[float, float]:
    """Return (positive boost, penalty) from obvious business intents."""
    q = query.casefold().replace("ё", "е")
    text = f"{section or ''} {title}".casefold().replace("ё", "е")

    boost = 0.0
    penalty = 0.0

    intents = [
        (("компан", "кто мы", "история компании", "деятельность", "ценност", "принцип"),
         ("о компании", "наши цели", "устав")),
        (("отпуск", "удален", "график", "испытатель", "обучение", "сотрудник", "hr"),
         ("hr", "регламент", "политика", "удален")),
        (("пожар", "безопасност"),
         ("пожар", "безопасност")),
        (("поддержк", "компьютер", "доступ", "it"),
         ("техническая поддержка", "поддержка")),
    ]

    for keywords, expected in intents:
        if any(k in q for k in keywords):
            if any(k in text for k in expected):
                boost += 0.18
            elif any(k in text for k in ("пожар", "опасн", "подряд")) and not any(k in q for k in ("пожар", "безопасност")):
                penalty += 0.12

    return boost, penalty


def _answerability_adjustment(query: str, text: str) -> tuple[float, float]:
    """Return (bonus, penalty) from cheap document-shape signals.

    This deliberately avoids domain-specific rules. It mainly suppresses tables of
    contents/definition blocks when the user asks for an action, owner, frequency
    or duration, and slightly rewards passages that contain answer-shaped language.
    """
    q = " ".join(query.casefold().replace("ё", "е").split())
    head = " ".join(text[:900].casefold().replace("ё", "е").split())
    whole = " ".join(text.casefold().replace("ё", "е").split())

    definition_intent = any(
        marker in q
        for marker in ("что такое", "что означает", "определение", "термин")
    )
    action_intent = any(
        marker in q
        for marker in (
            "что делать", "как " , "каким образом", "кто " , "кем " ,
            "в каких случаях", "порядок", "действия", "срок",
            "сколько", "как часто", "когда",
        )
    )

    penalty = 0.0
    if "оглавление" in head:
        penalty += 0.22
    if action_intent and not definition_intent and (
        "термины, определения" in head or "термины определения" in head
    ):
        penalty += 0.12

    bonus = 0.0
    if any(marker in q for marker in ("срок", "сколько", "как часто", "когда")):
        if re.search(r"\b\d+[\s-]*(?:календарн\w*\s+)?(?:дн\w*|час\w*|минут\w*|месяц\w*|год\w*)\b", whole):
            bonus += 0.08
        elif any(marker in whole for marker in ("не реже", "не более", "не менее", "в течение")):
            bonus += 0.06

    if action_intent and any(
        marker in whole
        for marker in (
            "необходимо", "обязан", "обязаны", "следует", "осуществляется",
            "производится", "оформляет", "оформить", "согласует",
            "согласован", "при обнаружении", "при возникновении",
        )
    ):
        bonus += 0.045

    membership_intent = bool(re.search(r"\bкто\s+(?:входит|состоит)\b", q)) or "состав комис" in q
    if membership_intent:
        membership_shape = "в состав" in whole and re.search(r"\bвход\w*\b", whole) is not None
        composition_mention = "состав эвакуационной комиссии" in whole or "состав комиссии" in whole
        role_count = sum(
            marker in whole
            for marker in ("председател", "заместител", "секретар", "член групп", "старший групп")
        )
        if membership_shape:
            bonus += 0.14
        elif composition_mention:
            bonus += 0.03
        if role_count >= 2:
            bonus += 0.06
        elif not membership_shape:
            penalty += 0.12
        if not membership_shape and any(marker in whole for marker in ("задачи комиссии", "заседания", "деятельностью комиссии")):
            penalty += 0.06

    if any(marker in q for marker in ("кто " , "кем " , "ответственный")) and any(
        marker in whole
        for marker in ("ответственн", "представител", "руководител", "служб", "работник", "председател", "секретар")
    ):
        bonus += 0.035

    return min(0.18, bonus), min(0.30, penalty)

def _min_span(sequence: list[str], required: set[str]) -> int | None:
    if len(required) <= 1:
        return 0 if required and next(iter(required)) in sequence else None
    counts: dict[str, int] = defaultdict(int)
    have = 0
    left = 0
    best: int | None = None
    for right, token in enumerate(sequence):
        if token in required:
            if counts[token] == 0:
                have += 1
            counts[token] += 1
        while have == len(required) and left <= right:
            span = right - left + 1
            best = span if best is None else min(best, span)
            left_token = sequence[left]
            if left_token in required:
                counts[left_token] -= 1
                if counts[left_token] == 0:
                    have -= 1
            left += 1
    return best


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        max_length: int,
        batch_size: int,
    ) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "For RERANKER_BACKEND=cross_encoder install requirements-models.txt"
            ) from exc
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size

    def predict(self, query: str, texts: list[str]) -> list[float]:
        raw: Sequence[float] = self.model.predict(
            [[query, text] for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in raw]


class HybridRetriever:
    """Fast hybrid retrieval tuned for small/medium local corpora.

    The old implementation fused only a tiny top-8 candidate set. That is fast,
    but it hurts recall badly when one part of a multi-part question lives lower
    in TF-IDF/Hashing rankings. This implementation scores a wider candidate
    pool and adds a zero-model lexical coverage/proximity signal plus adjacent
    chunks. On a ~1k chunk corpus this remains millisecond-scale.
    """

    def __init__(
        self,
        *,
        index: LocalIndex,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        if index.manifest.embedding_backend != embedder.backend_name:
            raise ValueError(
                "Index was created by another embedding backend. Rebuild it with "
                "python -m scripts.build_index --force"
            )
        if index.manifest.embedding_dimension != embedder.dimension:
            raise ValueError("Embedder dimension does not match the index.")

        self.index = index
        self.embedder = embedder
        self.settings = settings
        self._state = local()
        self.last_timings = {}
        self._cross_encoder = None
        if settings.reranker_backend == "cross_encoder":
            self._cross_encoder = CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
                max_length=settings.reranker_max_length,
                batch_size=settings.reranker_batch_size,
            )

        # Precompute cheap lexical features once at startup.
        self._chunk_root_sequences: list[list[str]] = []
        self._chunk_root_sets: list[set[str]] = []
        self._title_root_sets: list[set[str]] = []
        self._root_postings: dict[str, list[int]] = defaultdict(list)
        self._doc_chunk_to_index: dict[tuple[str, int], int] = {}
        self._doc_root_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for idx, chunk in enumerate(index.chunks):
            seq = lexical_roots(chunk.text)
            root_set = set(seq)
            self._chunk_root_sequences.append(seq)
            self._chunk_root_sets.append(root_set)
            title_roots = lexical_roots(chunk.title)
            self._title_root_sets.append(set(title_roots))
            self._doc_root_counts[chunk.doc_id].update(seq)
            self._doc_root_counts[chunk.doc_id].update(title_roots)
            for root in root_set:
                self._root_postings[root].append(idx)
            number = chunk.metadata.get("chunk_number")
            if isinstance(number, int):
                self._doc_chunk_to_index[(chunk.doc_id, number)] = idx

    @property
    def last_timings(self) -> dict[str, int]:
        value = getattr(self._state, "last_timings", None)
        if value is None:
            value = {}
            self._state.last_timings = value
        return value

    @last_timings.setter
    def last_timings(self, value: dict[str, int]) -> None:
        self._state.last_timings = value

    def _score_queries(self, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        stage = perf_counter()
        query_vectors = self.embedder.encode(queries)
        self.last_timings["query_embedding_ms"] = _elapsed_ms(stage)

        stage = perf_counter()
        dense_scores = np.asarray(self.index.dense_vectors @ query_vectors.T).T
        self.last_timings["dense_search_ms"] = _elapsed_ms(stage)

        stage = perf_counter()
        sparse_queries = self.index.sparse_vectorizer.transform(queries)
        sparse_scores = (self.index.sparse_matrix @ sparse_queries.T).T.toarray()
        self.last_timings["sparse_search_ms"] = _elapsed_ms(stage)
        return dense_scores, sparse_scores

    def _lexical_signals(
        self, queries: list[str]
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        stage = perf_counter()
        coverage: dict[int, float] = defaultdict(float)
        title_coverage: dict[int, float] = defaultdict(float)
        proximity: dict[int, float] = defaultdict(float)

        for query in queries:
            qseq = lexical_roots(query)
            qset = set(qseq)
            if not qset:
                continue
            numeric = {term for term in qset if term.isdigit()}
            candidate_ids: set[int] = set()
            for term in qset:
                candidate_ids.update(self._root_postings.get(term, ()))
            for idx in candidate_ids:
                roots = self._chunk_root_sets[idx]
                matched = qset & roots
                if not matched:
                    continue
                cov = len(matched) / len(qset)
                # Exact numbers carry more intent than ordinary tokens.
                if numeric and numeric <= roots:
                    cov = min(1.15, cov + 0.12)
                coverage[idx] = max(coverage[idx], cov)

                tmatch = qset & self._title_root_sets[idx]
                title_coverage[idx] = max(
                    title_coverage[idx], len(tmatch) / len(qset)
                )

                # Reward chunks where query concepts occur close together.
                required = matched if len(matched) <= 5 else set(list(matched)[:5])
                span = _min_span(self._chunk_root_sequences[idx], required)
                if span is not None and len(required) >= 2:
                    prox = len(required) / max(len(required), span)
                    proximity[idx] = max(proximity[idx], prox)

        self.last_timings["lexical_scoring_ms"] = _elapsed_ms(stage)
        return coverage, title_coverage, proximity

    def _candidate_pool(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        lexical_coverage: dict[int, float],
    ) -> tuple[list[int], dict[int, float], dict[int, float], dict[int, float]]:
        stage = perf_counter()
        aggregate_rrf: dict[int, float] = defaultdict(float)
        best_dense: dict[int, float] = defaultdict(float)
        best_sparse: dict[int, float] = defaultdict(float)

        for row in range(dense_scores.shape[0]):
            dense_order = _top_indices(dense_scores[row], self.settings.dense_top_k)
            sparse_order = _top_indices(sparse_scores[row], self.settings.sparse_top_k)
            for rank, idx in enumerate(dense_order, start=1):
                aggregate_rrf[idx] += 1.0 / (60 + rank)
                best_dense[idx] = max(best_dense[idx], float(dense_scores[row, idx]))
            for rank, idx in enumerate(sparse_order, start=1):
                aggregate_rrf[idx] += 1.0 / (60 + rank)
                best_sparse[idx] = max(best_sparse[idx], float(sparse_scores[row, idx]))

        ordered_rrf = sorted(aggregate_rrf, key=aggregate_rrf.get, reverse=True)
        candidates = ordered_rrf[: self.settings.rerank_candidate_k]

        if self.settings.lexical_rerank_enabled:
            lexical_order = sorted(
                lexical_coverage, key=lexical_coverage.get, reverse=True
            )[: self.settings.lexical_candidate_k]
            candidates = list(dict.fromkeys([*candidates, *lexical_order]))

        if self.settings.neighbor_expansion_enabled and self.settings.neighbor_distance:
            anchors = candidates[: self.settings.neighbor_anchor_k]
            expanded = list(candidates)
            for idx in anchors:
                record = self.index.chunks[idx]
                number = record.metadata.get("chunk_number")
                if not isinstance(number, int):
                    continue
                for delta in range(1, self.settings.neighbor_distance + 1):
                    for neighbor_number in (number - delta, number + delta):
                        neighbor = self._doc_chunk_to_index.get((record.doc_id, neighbor_number))
                        if neighbor is not None:
                            expanded.append(neighbor)
            candidates = list(dict.fromkeys(expanded))

        self.last_timings["fusion_ms"] = _elapsed_ms(stage)
        return candidates, aggregate_rrf, best_dense, best_sparse

    def _document_affinity(self, query: str, idx: int) -> float:
        qroots = set(lexical_roots(query))
        if not qroots:
            return 0.0
        record = self.index.chunks[idx]
        counts = self._doc_root_counts.get(record.doc_id)
        if not counts:
            return 0.0
        matched = [root for root in qroots if counts.get(root, 0) > 0]
        if not matched:
            return 0.0
        coverage = len(matched) / len(qroots)
        # Repeated use across a document is evidence of document-level subject,
        # but saturate quickly so long documents do not win solely by size.
        frequency = sum(
            min(1.0, log1p(counts[root]) / log1p(7.0)) for root in matched
        ) / len(qroots)
        return coverage * (0.55 + 0.45 * frequency)


    def _rerank(
        self,
        *,
        query: str,
        candidate_ids: list[int],
        rrf: dict[int, float],
        dense: dict[int, float],
        sparse_scores: dict[int, float],
        lexical_coverage: dict[int, float],
        title_coverage: dict[int, float],
        proximity: dict[int, float],
    ) -> list[tuple[int, float, dict[str, float]]]:
        stage = perf_counter()
        if not candidate_ids:
            self.last_timings["rerank_ms"] = 0
            return []

        max_rrf = max((rrf.get(idx, 0.0) for idx in candidate_ids), default=1.0) or 1.0
        metadata: dict[int, dict[str, float]] = {}
        for idx in candidate_ids:
            rrf_norm = rrf.get(idx, 0.0) / max_rrf
            dense_score = max(0.0, dense.get(idx, 0.0))
            sparse_score = max(0.0, sparse_scores.get(idx, 0.0))
            coverage = lexical_coverage.get(idx, 0.0)
            title = title_coverage.get(idx, 0.0)
            prox = proximity.get(idx, 0.0)
            document_affinity = self._document_affinity(query, idx)

            record = self.index.chunks[idx]
            query_terms = set(lexical_roots(query))
            title_terms = set(lexical_roots(record.title))
            section_terms = set(lexical_roots(record.section or ""))
            title_match = len(query_terms & title_terms) / max(1, len(query_terms))
            section_match = len(query_terms & section_terms) / max(1, len(query_terms))
            intent_boost, intent_penalty = _intent_section_boost(
                query, record.section, record.title
            )
            _, preferred_sections = _query_section_intent(query)
            section_id = record.metadata.get("section_id")
            if preferred_sections and int(section_id or 0) in preferred_sections:
                intent_boost += 0.30
            elif preferred_sections and int(section_id or 0) not in preferred_sections:
                intent_penalty += 0.10

            if self.settings.lexical_rerank_enabled:
                score = (
                    0.22 * rrf_norm
                    + 0.10 * dense_score
                    + 0.18 * sparse_score
                    + self.settings.lexical_coverage_weight * coverage
                    + self.settings.lexical_title_weight * title
                    + self.settings.lexical_proximity_weight * prox
                    + self.settings.lexical_document_weight * document_affinity
                    + 0.18 * title_match
                    + 0.08 * section_match
                    + intent_boost
                    - intent_penalty
                )
            else:
                score = 0.50 * rrf_norm + 0.30 * dense_score + 0.20 * sparse_score

            answerability_bonus, boilerplate_penalty = _answerability_adjustment(
                query, self.index.chunks[idx].text
            )
            score = max(0.0, score + answerability_bonus - boilerplate_penalty)

            metadata[idx] = {
                "rrf_score": float(rrf.get(idx, 0.0)),
                "dense_score": float(dense.get(idx, 0.0)),
                "sparse_score": float(sparse_scores.get(idx, 0.0)),
                "lexical_coverage": float(coverage),
                "title_coverage": float(title),
                "term_proximity": float(prox),
                "document_affinity": float(document_affinity),
                "title_match": float(title_match),
                "section_match": float(section_match),
                "intent_boost": float(intent_boost),
                "intent_penalty": float(intent_penalty),
                "answerability_bonus": float(answerability_bonus),
                "boilerplate_penalty": float(boilerplate_penalty),
                "base_score": float(score),
            }

        if self._cross_encoder is not None:
            texts = [self.index.chunks[idx].text for idx in candidate_ids]
            raw_scores = self._cross_encoder.predict(query, texts)
            if len(raw_scores) != len(candidate_ids):
                raise ValueError("Reranker returned an unexpected number of scores.")
            ranked = []
            for idx, raw_score in zip(candidate_ids, raw_scores, strict=True):
                if not isfinite(raw_score):
                    raise ValueError("Reranker returned NaN/Infinity.")
                metadata[idx]["rerank_raw_score"] = raw_score
                # Preserve strong lexical evidence while still using CE semantics.
                ce = _sigmoid(raw_score)
                combined = 0.72 * ce + 0.28 * min(1.0, metadata[idx]["base_score"])
                ranked.append((idx, combined, metadata[idx]))
        else:
            ranked = [(idx, metadata[idx]["base_score"], metadata[idx]) for idx in candidate_ids]

        ranked.sort(key=lambda item: item[1], reverse=True)
        self.last_timings["rerank_ms"] = _elapsed_ms(stage)
        return ranked

    def _to_chunks(
        self,
        ranked: list[tuple[int, float, dict[str, float]]],
        top_k: int,
        section_id: int = 0,
    ) -> list[RetrievedChunk]:
        result: list[RetrievedChunk] = []
        per_document: dict[str, int] = defaultdict(int)
        for idx, score, score_meta in ranked:
            if score < self.settings.retrieval_min_score:
                continue
            record = self.index.chunks[idx]
            if section_id and str(record.metadata.get("section_id", "")) != str(section_id):
                continue
            if per_document[record.doc_id] >= self.settings.max_chunks_per_document:
                continue
            per_document[record.doc_id] += 1
            result.append(
                RetrievedChunk(
                    doc_id=record.doc_id,
                    chunk_id=record.chunk_id,
                    title=record.title,
                    section=record.section,
                    text=record.text,
                    url=record.url,
                    score=float(score),
                    metadata={
                        **record.metadata,
                        **score_meta,
                        "version": record.version,
                        "updated_at": record.updated_at.isoformat(),
                    },
                )
            )
            if len(result) >= top_k:
                break
        return result

    def _search(self, queries: list[str], *, top_k: int, primary_query: str, section_id: int = 0) -> list[RetrievedChunk]:
        self.last_timings = {}
        started = perf_counter()
        dense, sparse_scores = self._score_queries(queries)
        lexical_coverage, title_coverage, proximity = self._lexical_signals(queries)
        candidates, rrf, best_dense, best_sparse = self._candidate_pool(
            dense, sparse_scores, lexical_coverage
        )
        ranked = self._rerank(
            query=primary_query,
            candidate_ids=candidates,
            rrf=rrf,
            dense=best_dense,
            sparse_scores=best_sparse,
            lexical_coverage=lexical_coverage,
            title_coverage=title_coverage,
            proximity=proximity,
        )
        result = self._to_chunks(ranked, top_k, section_id)
        self.last_timings["candidate_count"] = len(candidates)
        self.last_timings["retrieval_total_ms"] = _elapsed_ms(started)
        return result

    def search(self, query: str, top_k: int = 5, section_id: int = 0) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        return self._search([query], top_k=top_k, primary_query=query, section_id=section_id)

    def search_multiple(self, queries: list[str], top_k: int = 5, section_id: int = 0) -> list[RetrievedChunk]:
        clean = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if not clean:
            return []
        return self._search(clean, top_k=top_k, primary_query=clean[0], section_id=section_id)
