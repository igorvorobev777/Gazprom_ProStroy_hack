from __future__ import annotations

import re
from dataclasses import dataclass

from .fast_context import roots
from .schemas import RetrievedChunk

_CITATION = re.compile(r"\[S(\d+)\]", re.I)
_PLACEHOLDER = re.compile(r"\[(?:ОРГАНИЗАЦИЯ|ЧЕЛОВЕК|АДРЕС)_\d+\]", re.I)
_URL = re.compile(r"https?://\S+", re.I)


@dataclass(frozen=True, slots=True)
class GroundingAssessment:
    valid: bool
    min_claim_coverage: float
    checked_claims: int
    issues: tuple[str, ...]


def _numbers(text: str) -> set[str]:
    # Ignore one-digit list numbering, but keep dates, phone fragments, durations.
    return set(re.findall(r"\b\d{2,}\b", text))


def _latin_entities(text: str) -> set[str]:
    return {x.casefold() for x in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", text)}


def assess_answer_grounding(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    min_claim_coverage: float = 0.18,
) -> GroundingAssessment:
    source_map = {c.source_id: c for c in chunks if c.source_id}
    issues: list[str] = []
    coverages: list[float] = []
    checked = 0

    if _URL.search(answer):
        issues.append("answer_contains_url")

    # Treat a citation marker as the end of a factual unit. This supports both
    # "claim [S1]." and "claim. [S1]" without accidentally marking the claim
    # uncited merely because punctuation comes before the citation.
    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r"\[S\d+\](?:[.!?])?", answer, flags=re.I):
        part = answer[cursor:match.end()].strip()
        if part:
            parts.append(part)
        cursor = match.end()
    tail = answer[cursor:].strip()
    if tail:
        parts.append(tail)

    for part in parts:
        clean = _CITATION.sub("", part)
        clean = _PLACEHOLDER.sub(" ", clean)
        clean = re.sub(r"^\s*\d+[.)]\s*", "", clean).strip(" -—:;,.\t")
        claim_roots = roots(clean)
        if not claim_roots:
            continue
        checked += 1
        ids = [f"S{n}" for n in _CITATION.findall(part)]
        if not ids:
            issues.append("uncited_claim")
            continue

        cited = [source_map[source_id] for source_id in ids if source_id in source_map]
        if not cited:
            issues.append("unknown_citation")
            continue
        source_text = " ".join(f"{c.title} {c.section or ''} {c.text}" for c in cited)
        source_roots = roots(source_text)
        coverage = len(claim_roots & source_roots) / max(1, len(claim_roots))
        coverages.append(coverage)

        # Numeric and explicit latin tokens are high-risk hallucination anchors.
        if not _numbers(clean) <= _numbers(source_text):
            issues.append("unsupported_number")
        if not _latin_entities(clean) <= _latin_entities(source_text):
            issues.append("unsupported_latin_entity")
        if coverage < min_claim_coverage:
            issues.append("low_claim_source_overlap")

    if checked == 0:
        issues.append("no_factual_claims")
    return GroundingAssessment(
        valid=not issues,
        min_claim_coverage=min(coverages, default=0.0),
        checked_claims=checked,
        issues=tuple(dict.fromkeys(issues)),
    )
