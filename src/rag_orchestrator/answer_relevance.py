from __future__ import annotations

import re
from dataclasses import dataclass

from .fast_context import roots
from .query_planner import specific_roots

_PLACEHOLDER = re.compile(r"\[(?:ОРГАНИЗАЦИЯ|ЧЕЛОВЕК|АДРЕС)_\d+\]", re.I)

_ACRONYM_MEANINGS = {
    "соут": ("специальная", "оценка", "условий", "труда"),
    "сиз": ("средства", "индивидуальной", "защиты"),
}


@dataclass(frozen=True, slots=True)
class RelevanceAssessment:
    valid: bool
    issues: tuple[str, ...]


def assess_answer_relevance(query: str, answer: str) -> RelevanceAssessment:
    q = " ".join(query.casefold().replace("ё", "е").split())
    a = _PLACEHOLDER.sub(" ", answer.casefold().replace("ё", "е"))
    aroots = roots(a)
    issues: list[str] = []

    # Preserve the object of the question when it is explicit. For known acronyms,
    # accepting their expanded form avoids rejecting a better explanatory answer.
    anchors = set(specific_roots(query))
    if anchors:
        overlap = anchors & aroots
        acronym_supported = False
        for acronym, meaning in _ACRONYM_MEANINGS.items():
            if re.search(rf"\b{acronym}\b", q):
                meaning_roots = roots(" ".join(meaning))
                acronym_supported = bool(meaning_roots & aroots)
                break
        if not overlap and not acronym_supported:
            issues.append("answer_misses_query_object")

    who_intent = bool(re.search(r"\b(кто|кому|кем)\b", q)) or "ответствен" in q
    time_intent = any(marker in q for marker in ("срок", "сколько", "как часто", "периодич", "когда"))
    definition_intent = any(marker in q for marker in ("что такое", "что означает", "определение"))
    membership_intent = bool(re.search(r"\bкто\s+(?:входит|состоит)\b", q)) or "состав комис" in q
    procedure_intent = any(
        marker in q
        for marker in (
            "что делать", "как оформ", "как провод", "как выполня", "порядок",
            "какие действия", "какие меры", "какие требования", "в каких случаях",
        )
    )

    if who_intent:
        if not re.search(
            r"\b(?:ответствен\w*|руководител\w*|представител\w*|работник\w*|"
            r"комисси\w*|служб\w*|организаци\w*|лиц\w*|директор\w*|специалист\w*|"
            r"диспетчер\w*|подрядчик\w*)\b",
            a,
        ):
            issues.append("answer_misses_who_intent")

    if membership_intent:
        if not re.search(
            r"\b(?:входит\w*|состо\w*|член\w*|председател\w*|"
            r"заместител\w*|секретар\w*)\b",
            a,
        ):
            issues.append("answer_misses_membership_intent")

    if "соглас" in q and "соглас" not in a:
        issues.append("answer_misses_agreement_action")
    if "сообщ" in q and "сообщ" not in a:
        issues.append("answer_misses_notification_action")

    if definition_intent:
        definition_shape = bool(
            re.search(r"(?:—|-)\s*это\b|\bэто\b|представляет собой|определяется как|понимается как", a)
        )
        if "соут" in q:
            expanded = all(root in aroots for root in roots("специальная оценка условий труда"))
            definition_shape = definition_shape or expanded
        if not definition_shape:
            issues.append("answer_misses_definition_intent")

    if time_intent:
        if not (
            re.search(r"\b\d+(?:[.,]\d+)?\b", a)
            or any(marker in a for marker in ("не более", "не менее", "в течение", "ежегод", "ежеднев", "ежемесяч", "по истечении"))
        ):
            issues.append("answer_misses_time_intent")

    if procedure_intent:
        if not any(
            marker in a
            for marker in (
                "необходимо", "следует", "обязан", "обязаны", "оформ", "соглас",
                "сообщ", "провод", "выполн", "назнач", "провер", "обеспеч", "запрещ",
                "допускается", "эваку", "принять меры",
            )
        ):
            issues.append("answer_misses_procedure_intent")

    return RelevanceAssessment(valid=not issues, issues=tuple(dict.fromkeys(issues)))
