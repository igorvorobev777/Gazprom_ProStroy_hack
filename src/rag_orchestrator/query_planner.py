from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.I)

_STOP = {
    "а", "без", "бы", "был", "была", "были", "быть", "в", "вам", "вас", "во",
    "вот", "все", "для", "до", "его", "ее", "её", "если", "есть", "же", "за", "и",
    "из", "или", "им", "их", "к", "как", "какая", "какие", "какой", "кем", "когда",
    "кто", "кому", "ли", "на", "над", "не", "нет", "но", "о", "об", "он", "она",
    "они", "оно", "от", "по", "под", "при", "про", "с", "со", "так", "то", "у",
    "что", "чем", "чего", "это", "эта", "эти", "этот", "почему", "зачем", "куда", "где", "расскажи",
    "напиши", "покажи", "объясни", "скажи", "пожалуйста",
}

# These words describe the shape of a question, not its object. A short query made
# only of such roots is under-specified even if retrieval finds many lexical hits.
_GENERIC_ROOTS = {
    "ответс", "отвеч", "оформ", "делат", "дейст", "нужн", "докуме", "меры", "безопа",
    "срок", "услов", "измене", "соглас", "провод", "провед", "выполн", "работ", "поряд",
    "требов", "долже", "сообщ", "входи", "дейст", "случа", "вопрос", "информ",
    "нужны", "нужен", "нужна", "нужно", "провер", "звони", "звонит", "продл",
    "допуск", "допуска", "увиде", "обнару",
}

_DIRECT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"^(?:привет|здравствуй(?:те)?|доброе\s+утро|добрый\s+день|добрый\s+вечер)[!.?\s]*$", re.I),
        "Здравствуйте! Я готов помочь с вопросами по базе знаний.",
        "smalltalk_greeting",
    ),
    (
        re.compile(r"^(?:как\s+дела|как\s+ты|что\s+нового|как\s+настроение|как\s+поживаешь)[!.?\s]*$", re.I),
        "Всё хорошо. Я готов помочь с вопросами по базе знаний.",
        "smalltalk_status",
    ),
    (
        re.compile(r"^(?:(?:привет|здравствуй(?:те)?|добрый\s+(?:день|вечер)|доброе\s+утро)[,!]?\s*)?(?:как\s+дела|как\s+ты|как\s+настроение|что\s+нового)[!.?\s]*$", re.I),
        "Всё хорошо. Я готов помочь с вопросами по базе знаний.",
        "smalltalk_status",
    ),
    (
        re.compile(r"^(?:(?:ок|хорошо|понятно)[,!]?\s*)?(?:спасибо(?:\s+большое)?|благодарю|спс)[!.?\s]*$", re.I),
        "Пожалуйста. Если нужно, задайте следующий вопрос по базе знаний.",
        "smalltalk_thanks",
    ),
    (
        re.compile(r"^(?:пока|до\s+свидания|до\s+встречи)[!.?\s]*$", re.I),
        "До свидания!",
        "smalltalk_bye",
    ),
    (
        re.compile(r"^(?:что\s+ты\s+умеешь|что\s+умеешь|чем\s+можешь\s+помочь|какие\s+вопросы\s+можно\s+задавать)[!.?\s]*$", re.I),
        "Я отвечаю на вопросы по подключённой базе знаний HiHub и показываю источники ответа.",
        "meta_capabilities",
    ),
    (
        re.compile(r"^(?:помоги|нужна\s+помощь|помощь)[!.?\s]*$", re.I),
        "Задайте конкретный вопрос по материалам базы знаний — я найду релевантные источники и отвечу по ним.",
        "meta_help",
    ),
)

_ACRONYM_EXPANSIONS = {
    "соуt": "специальная оценка условий труда",  # defensive latin t typo
    "соут": "специальная оценка условий труда",
    "сиз": "средства индивидуальной защиты",
    "нд": "наряд-допуск",
    "рпо": "работы повышенной опасности",
}


def _root(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    if token.isdigit():
        return token
    if len(token) >= 9:
        return token[:6]
    if len(token) >= 6:
        return token[:5]
    return token


def content_roots(query: str) -> tuple[str, ...]:
    roots: list[str] = []
    for raw in _TOKEN_RE.findall(query.casefold().replace("ё", "е")):
        if raw in _STOP or (len(raw) < 3 and not raw.isdigit()):
            continue
        value = _root(raw)
        if value not in roots:
            roots.append(value)
    return tuple(roots)


def specific_roots(query: str) -> tuple[str, ...]:
    return tuple(root for root in content_roots(query) if root not in _GENERIC_ROOTS)


@dataclass(frozen=True, slots=True)
class DirectDecision:
    answer: str
    reason: str


def direct_response(query: str) -> DirectDecision | None:
    normalized = " ".join(query.strip().split())
    for pattern, answer, reason in _DIRECT_PATTERNS:
        if pattern.fullmatch(normalized):
            return DirectDecision(answer=answer, reason=reason)
    return None


def clarification_for_underspecified(query: str) -> str | None:
    """Ask for an object when a short query contains only generic action words.

    Retrieval cannot fix missing intent. Queries such as ``Кто отвечает?`` or
    ``Как оформить?`` match hundreds of clauses in regulatory documents and must
    not be converted into a confident random answer.
    """
    q = " ".join(query.casefold().replace("ё", "е").split())
    roots = content_roots(query)
    anchors = specific_roots(query)

    # Explicit contextual/anaphoric references are unsafe in the stateless UI.
    anaphoric = bool(re.search(r"\b(их|это|эти|этого|такое|такие|ему|ей|им)\b", q))
    if anaphoric and not anchors:
        return "Уточните, пожалуйста, к какому документу, процессу или виду работ относится вопрос."

    # Only apply to short/generic queries. Long questions can legitimately use
    # broad words while specifying the object in a phrase not covered by our list.
    if roots and not anchors and len(_TOKEN_RE.findall(q)) <= 7:
        return (
            "Уточните, пожалуйста, объект вопроса: какой документ, процесс, "
            "вид работ или ситуация вас интересует."
        )
    return None


_EXTERNAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:погода|прогноз\s+погоды|температур\w*\s+(?:на\s+улице|снаружи))\b", re.I), "external_weather"),
    (re.compile(r"\bкурс\s+(?:доллар\w*|евро|валют\w*)\b", re.I), "external_currency"),
    (re.compile(r"\b(?:рецепт|приготовить|как\s+готовить)\b", re.I), "external_recipe"),
    (re.compile(r"\b(?:матч|счет\s+матча|результат\s+матча)\b", re.I), "external_sports"),
    (re.compile(r"\b(?:кто\s+президент|президент\s+[А-ЯA-Z])", re.I), "external_current_affairs"),
    (re.compile(r"(?:\bсколько\s+будет\b.*\d|\d\s*[+*/=]\s*\d)", re.I), "external_arithmetic"),
)


def external_query_reason(query: str) -> str | None:
    """Recognize a few high-precision intents that cannot be answered by this KB.

    This is intentionally conservative: it is a domain boundary, not a general
    topic classifier. Ambiguous queries continue to the evidence gate.
    """
    for pattern, reason in _EXTERNAL_PATTERNS:
        if pattern.search(query):
            return reason
    return None


def _strip_polite_prefixes(query: str) -> str:
    text = " ".join(query.strip().split())
    # Remove only leading conversational wrappers. The substantive question is
    # left untouched, which keeps retrieval deterministic and easy to audit.
    prefix = re.compile(
        r"^(?:(?:привет|здравствуй(?:те)?|доброе\s+утро|добрый\s+(?:день|вечер))[,!]?\s*)?"
        r"(?:(?:подскажи(?:те)?|скажи(?:те)?|расскажи(?:те)?|объясни(?:те)?)[,!]?\s*)?"
        r"(?:(?:пожалуйста)[,!]?\s*)?",
        re.I,
    )
    cleaned = prefix.sub("", text).strip(" ,")
    return cleaned or text


def normalize_search_query(query: str) -> str:
    """Normalize conversational wording without broad semantic expansion."""
    normalized = _strip_polite_prefixes(query)
    normalized = re.sub(r"(?iu)\bнаряд\s+допуск\b", "наряд-допуск", normalized)
    # Colloquial formulation used for work permits in this corpus.
    if "наряд-допуск" not in normalized.casefold() and re.search(
        r"(?iu)\bнаряд\b.*\b(?:опасн\w*|повышенн\w*)\b", normalized
    ):
        normalized = re.sub(r"(?iu)\bнаряд\b", "наряд-допуск", normalized, count=1)
    normalized = re.sub(r"(?iu)\bвыпис(?:ать|ывают|ывается)\b", "оформить", normalized)
    if "наряд-допуск" in normalized.casefold():
        normalized = re.sub(
            r"(?iu)\b(?:опасн\w*\s+работ\w*|работ\w*\s+опасн\w*)\b",
            "работы повышенной опасности",
            normalized,
        )

    q = normalized.casefold().replace("ё", "е")
    # Colloquial emergency wording is normalized to terminology that appears in
    # the fire instructions. This is intentionally limited to action questions.
    if (
        re.search(r"\b(?:огонь|огня|дым|дыма)\b", q)
        and re.search(r"(?:что\s+.*(?:делать|сделать)|если\s+(?:увидел|обнаружил)|увидел|обнаружил)", q)
    ):
        return "Что делать при обнаружении пожара признаков горения задымления?"
    if re.search(r"\bвозгоран\w*\b", q):
        normalized = f"{normalized} пожар"
    return normalized


def build_search_queries(query: str) -> list[str]:
    """Canonicalize spelling without broadening intent.

    Earlier experiments expanded acronyms (for example SIZ -> full phrase). On the
    real HiHub corpus that occasionally promoted a different regulation. Exact
    acronyms are already indexed, so the safest portable strategy is one query.
    """
    return [normalize_search_query(query)]


def clarification_from_retrieval(
    query: str,
    chunks: list,
) -> str | None:
    """Detect multiple equally plausible procedures after retrieval.

    A high score is not enough when several different documents are tied and none
    of their titles identifies the requested object. This is typical for generic
    ``наряд-допуск`` or ``СИЗ`` questions where the answer depends on work type.
    """
    if len(chunks) < 2 or not specific_roots(query):
        return None
    first, second = chunks[0], chunks[1]
    if first.doc_id == second.doc_id or first.score <= 0:
        return None
    if second.score / first.score < 0.94:
        return None
    top = chunks[:3]
    max_title = max(float(c.metadata.get("title_coverage", 0.0) or 0.0) for c in top)
    if max_title >= 0.15:
        return None
    return (
        "В базе найдено несколько одинаково релевантных процедур. "
        "Уточните вид работ, документ или ситуацию, к которой относится вопрос."
    )


def is_comparison_query(query: str) -> bool:
    q = query.casefold().replace("ё", "е")
    return any(marker in q for marker in ("сравни", "разниц", "отлич", "в разных документах", "между"))
