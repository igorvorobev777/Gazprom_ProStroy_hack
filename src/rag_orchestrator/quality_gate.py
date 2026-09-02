from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.I)
_STOP_WORDS = {
    "а", "без", "бы", "был", "была", "были", "быть", "в", "вам", "вас", "во", "вот", "все",
    "всех", "где", "да", "для", "до", "его", "ее", "её", "если", "есть", "еще", "ещё", "же", "за",
    "зачем", "и", "из", "или", "им", "их", "к", "как", "какая", "какие", "какой", "кем", "когда",
    "кто", "кому", "ли", "на", "над", "надо", "не", "него", "нее", "ней", "нет", "но", "о", "об",
    "он", "она", "они", "оно", "от", "по", "под", "при", "про", "с", "со", "так", "также", "такое",
    "такой", "то", "у", "уже", "что", "чем", "чего", "это", "эта", "эти", "этот", "должен", "должна",
    "должны", "нужно", "можно", "делать", "порядок", "каков", "какова", "напиши", "расскажи", "объясни",
}


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    level: Literal["weak", "borderline", "strong"]
    score: float
    exact_coverage: float
    stem_coverage: float
    sparse_score: float
    dense_score: float
    title_coverage: float
    top_retrieval_score: float
    title_support: float
    lexical_support: float
    document_coherence: float
    hard_anchor_missing: bool
    query_terms: tuple[str, ...]
    reason: str


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text.casefold().replace("ё", "е")):
        if raw in _STOP_WORDS:
            continue
        if len(raw) < 3 and not raw.isdigit():
            continue
        tokens.add(raw)
    return tokens


def _hard_anchors(query: str) -> set[str]:
    anchors: set[str] = set()
    # Latin product/entity names, explicit acronyms and multi-digit numbers are
    # high-information terms. If retrieval does not contain them, a match on one
    # generic word must not make the query answerable.
    anchors.update(token.casefold() for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", query))
    anchors.update(token.casefold().replace("ё", "е") for token in re.findall(r"\b[А-ЯЁA-Z]{2,8}\b", query))
    anchors.update(re.findall(r"\b\d{2,}\b", query))
    # A title-cased entity inside a sentence is usually a named entity (Москва,
    # Пушкин, Windows transliterations). If it is absent from the KB evidence, a
    # match on generic words must not make the query answerable. Skip the first
    # token because normal Russian sentences are capitalized there.
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9_-]+", query)
    for token in words[1:]:
        if len(token) >= 4 and token[:1].isupper() and not token.isupper():
            anchors.add(token.casefold().replace("ё", "е"))
    return anchors


def assess_evidence(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> EvidenceAssessment:
    if not chunks:
        return EvidenceAssessment(
            level="weak", score=0.0, exact_coverage=0.0, stem_coverage=0.0,
            sparse_score=0.0, dense_score=0.0, title_coverage=0.0,
            top_retrieval_score=0.0, title_support=0.0, lexical_support=0.0,
            document_coherence=0.0, hard_anchor_missing=False, query_terms=(),
            reason="no_chunks",
        )

    top = chunks[:3]
    query_terms = _tokens(query)
    exact_coverages: list[float] = []
    stem_values: list[float] = []
    sparse_values: list[float] = []
    dense_values: list[float] = []
    title_values: list[float] = []
    source_text = " ".join(f"{c.title} {c.text}" for c in chunks[:5]).casefold().replace("ё", "е")

    for chunk in top:
        source_terms = _tokens(f"{chunk.title} {chunk.text}")
        exact_coverages.append(
            len(query_terms & source_terms) / len(query_terms) if query_terms else 0.0
        )
        stem_values.append(float(chunk.metadata.get("lexical_coverage", 0.0) or 0.0))
        sparse_values.append(float(chunk.metadata.get("sparse_score", 0.0) or 0.0))
        dense_values.append(float(chunk.metadata.get("dense_score", 0.0) or 0.0))
        title_values.append(float(chunk.metadata.get("title_coverage", 0.0) or 0.0))

    exact_coverage = max(exact_coverages, default=0.0)
    stem_coverage = max(stem_values, default=0.0)
    sparse_score = max(sparse_values, default=0.0)
    dense_score = max(dense_values, default=0.0)
    title_coverage = max(title_values, default=0.0)
    top_score = max(0.0, float(chunks[0].score))
    title_support = sum(value > 0.0 for value in title_values) / max(1, len(title_values))
    lexical_support = sum(value >= 0.5 for value in stem_values) / max(1, len(stem_values))

    doc_counts: dict[str, int] = {}
    for chunk in top:
        doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
    document_coherence = max(doc_counts.values(), default=0) / max(1, len(top))

    hard = _hard_anchors(query)
    hard_anchor_missing = any(anchor not in source_text for anchor in hard)

    score = (
        0.22 * min(1.0, exact_coverage)
        + 0.14 * min(1.0, stem_coverage)
        + 0.14 * min(1.0, sparse_score / 0.12)
        + 0.08 * min(1.0, dense_score / 0.35)
        + 0.12 * min(1.0, title_coverage)
        + 0.08 * min(1.0, top_score / 0.75)
        + 0.10 * title_support
        + 0.08 * lexical_support
        + 0.04 * document_coherence
    )

    reason = "score"
    if hard_anchor_missing:
        score = min(score, weak_threshold - 0.01)
        reason = "missing_hard_anchor"

    # Typical lexical collision: one common word matches strongly in TF-IDF while
    # the query has no title support and no consistent lexical evidence.
    collision_like = (
        exact_coverage < 0.50
        and title_coverage < 0.10
        and lexical_support == 0.0
        and top_score < 0.40
    )
    if collision_like:
        score = min(score, weak_threshold - 0.01)
        reason = "collision_like"

    if score < weak_threshold:
        level: Literal["weak", "borderline", "strong"] = "weak"
    elif score < strong_threshold:
        level = "borderline"
    else:
        level = "strong"

    return EvidenceAssessment(
        level=level,
        score=max(0.0, min(1.0, score)),
        exact_coverage=exact_coverage,
        stem_coverage=stem_coverage,
        sparse_score=sparse_score,
        dense_score=dense_score,
        title_coverage=title_coverage,
        top_retrieval_score=top_score,
        title_support=title_support,
        lexical_support=lexical_support,
        document_coherence=document_coherence,
        hard_anchor_missing=hard_anchor_missing,
        query_terms=tuple(sorted(query_terms)),
        reason=reason,
    )
