from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.I)
_STOP = {
    "а", "без", "бы", "в", "во", "для", "до", "его", "ее", "её", "и", "из",
    "или", "их", "как", "какая", "какие", "какой", "кем", "когда", "кто", "ли",
    "на", "над", "не", "но", "о", "об", "от", "по", "под", "при", "про", "с",
    "со", "то", "у", "что", "это", "этот", "эта", "эти", "должен", "должна",
    "должны", "нужно", "можно", "порядок",
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


def roots(text: str) -> set[str]:
    return {
        _root(tok)
        for tok in _TOKEN_RE.findall(text.casefold().replace("ё", "е"))
        if tok not in _STOP and (len(tok) >= 3 or tok.isdigit())
    }


def query_acronyms(text: str) -> set[str]:
    tokens = [
        tok.casefold().replace("ё", "е")
        for tok in _TOKEN_RE.findall(text)
        if tok.casefold().replace("ё", "е") not in _STOP and len(tok) >= 3 and not tok.isdigit()
    ]
    result: set[str] = set()
    for width in range(3, min(5, len(tokens)) + 1):
        for start in range(0, len(tokens) - width + 1):
            acronym = "".join(token[0] for token in tokens[start : start + width])
            if len(acronym) >= 3:
                result.add(acronym)
    return result


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = [item.strip() for item in _SENTENCE_SPLIT.split(normalized) if item.strip()]
    return parts or ([normalized] if normalized else [])


def _intent_adjustment(query: str, sentence: str) -> float:
    q = " ".join(query.casefold().replace("ё", "е").split())
    s = " ".join(sentence.casefold().replace("ё", "е").split())
    bonus = 0.0

    who_intent = any(marker in q for marker in ("кто " , "кем " , "какая организация", "какое подразделение"))
    time_intent = any(marker in q for marker in ("срок", "сколько", "как часто", "когда", "периодич"))
    action_intent = any(marker in q for marker in ("что делать", "как " , "каким образом", "действия", "порядок", "в каких случаях"))
    definition_intent = any(marker in q for marker in ("что такое", "что означает", "определение", "термин"))
    membership_intent = bool(re.search(r"\bкто\s+(?:входит|состоит)\b", q)) or "состав комис" in q

    if membership_intent:
        membership_shape = (
            ("в состав" in s and re.search(r"\bвход\w*\b", s) is not None)
            or "состав эвакуационной комиссии" in s
        )
        role_count = sum(
            marker in s
            for marker in ("председател", "заместител", "секретар", "член групп", "старший групп")
        )
        if membership_shape:
            bonus += 0.65
        if role_count >= 2:
            bonus += 0.32
        if not membership_shape and any(marker in s for marker in ("задачи комиссии", "проводит плановые заседания", "руководство деятельностью")):
            bonus -= 0.35

    if who_intent:
        role_text = re.sub(r"\[[^\]]+\]", " ", s)
        has_role = bool(
            re.search(
                r"\b(?:эксперт(?:ом|а|ы|ов|ами)?|работник(?:ами|и|ов|ом)?|комисси\w*|руководител\w*|представител\w*|служб\w*|ответственн\w*)\b",
                role_text,
            )
            or re.search(r"организаци\w*[, ]+(?:осуществля|проводящ|выполняющ)\w*", role_text)
        )
        if has_role:
            bonus += 0.42
        else:
            bonus -= 0.42
        if any(marker in s for marker in ("проводится", "проводит", "осуществля", "выполняется", "принимается")):
            bonus += 0.16

    if time_intent:
        if re.search(r"\b\d+[\s-]*(?:календарн\w*\s+)?(?:дн\w*|час\w*|минут\w*|месяц\w*|год\w*)\b", s):
            bonus += 0.30
        elif any(marker in s for marker in ("не реже", "не более", "не менее", "в течение", "по истечении", "ежеднев", "ежегод")):
            bonus += 0.24

    if action_intent:
        if any(marker in s for marker in ("необходимо", "обязан", "обязаны", "следует", "немедленно", "незамедлительно", "осуществляется", "производится", "оформить", "оформляет", "согласует", "принять меры", "привести в действие", "эвакуац")):
            bonus += 0.20
        if "обязанности и действия" in s and len(sentence.strip()) < 140:
            bonus -= 0.22

    # Definition blocks often repeat all query terms but do not answer action,
    # owner or timing questions. Suppress them unless the user asks for a definition.
    if not definition_intent and (who_intent or time_intent or action_intent):
        if "термины, определения" in s or (" — " in s and any(marker in s for marker in ("документ", "состояние", "процесс", "комплекс положений"))):
            bonus -= 0.18
    if len(sentence.strip()) < 70 and re.search(r"(?:порядок|раздел|глава|приложение)\b", s):
        bonus -= 0.10
    return bonus


def _sentence_score(query_roots: set[str], sentence: str, query: str = "") -> float:
    sroots = roots(sentence)
    if not sroots or not query_roots:
        return 0.0
    overlap = len(query_roots & sroots)
    coverage = overlap / len(query_roots)
    density = overlap / max(1, len(sroots))
    acronym_bonus = 0.18 if query_acronyms(query) & sroots else 0.0
    return (
        0.82 * coverage
        + 0.18 * min(1.0, density * 4.0)
        + acronym_bonus
        + _intent_adjustment(query, sentence)
    )


def focus_chunk(
    query: str,
    chunk: RetrievedChunk,
    *,
    max_chars: int,
    neighbor_sentences: int,
) -> RetrievedChunk:
    """Select the most query-relevant sentence window without another model."""
    sentences = _sentences(chunk.text)
    if not sentences or len(chunk.text) <= max_chars:
        return chunk.model_copy(deep=True)

    qnorm = " ".join(query.casefold().replace("ё", "е").split())
    membership_intent = bool(re.search(r"\bкто\s+(?:входит|состоит)\b", qnorm)) or "состав комис" in qnorm
    if membership_intent:
        anchor_idx = next(
            (
                idx
                for idx, sentence in enumerate(sentences)
                if "в состав" in sentence.casefold().replace("ё", "е")
                and re.search(r"\bвход\w*\b", sentence.casefold().replace("ё", "е"))
            ),
            None,
        )
        if anchor_idx is not None:
            start = anchor_idx
            if anchor_idx > 0 and "состав" in sentences[anchor_idx - 1].casefold().replace("ё", "е"):
                start = anchor_idx - 1
            selected: list[str] = []
            role_hits = 0
            for sentence in sentences[start:]:
                candidate = " ".join([*selected, sentence]).strip()
                if selected and len(candidate) > max_chars:
                    break
                selected.append(sentence)
                role_hits += sum(
                    marker in sentence.casefold().replace("ё", "е")
                    for marker in ("председател", "заместител", "секретар", "старший групп", "член групп")
                )
                if len(" ".join(selected)) >= max_chars:
                    break
                if role_hits >= 4:
                    break
            text = " ".join(selected).strip()
            if text:
                metadata = dict(chunk.metadata)
                metadata.update(
                    {
                        "focused_passage": True,
                        "original_text_chars": len(chunk.text),
                        "selected_text_chars": len(text),
                        "focused_membership_block": True,
                    }
                )
                return chunk.model_copy(deep=True, update={"text": text, "metadata": metadata})

    qroots = roots(query)
    scored = [(_sentence_score(qroots, sentence, query), idx) for idx, sentence in enumerate(sentences)]
    scored.sort(reverse=True)
    best_idx = scored[0][1]
    start = max(0, best_idx - neighbor_sentences)
    end = min(len(sentences), best_idx + neighbor_sentences + 1)
    selected = sentences[start:end]

    # Grow around the best window while there is room. This keeps definitions,
    # conditions and the following responsible-party clause together.
    left, right = start - 1, end
    while True:
        options: list[tuple[float, str, int]] = []
        if left >= 0:
            options.append((_sentence_score(qroots, sentences[left], query), "left", left))
        if right < len(sentences):
            options.append((_sentence_score(qroots, sentences[right], query), "right", right))
        if not options:
            break
        options.sort(reverse=True)
        _, side, idx = options[0]
        candidate = ([sentences[idx]] + selected) if side == "left" else (selected + [sentences[idx]])
        text = " ".join(candidate)
        if len(text) > max_chars:
            break
        selected = candidate
        if side == "left":
            left -= 1
        else:
            right += 1

    text = " ".join(selected).strip()
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "focused_passage": True,
            "original_text_chars": len(chunk.text),
            "selected_text_chars": len(text),
            "focused_sentence_score": float(scored[0][0]),
        }
    )
    return chunk.model_copy(deep=True, update={"text": text, "metadata": metadata})


def prepare_context(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    focused: bool,
    passage_chars: int,
    neighbor_sentences: int,
    max_context_chars: int,
    max_chunks: int = 5,
    duplicate_jaccard: float = 0.88,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    selected_root_sets: list[set[str]] = []
    used_chars = 0
    for chunk in chunks:
        current = (
            focus_chunk(
                query,
                chunk,
                max_chars=passage_chars,
                neighbor_sentences=neighbor_sentences,
            )
            if focused
            else chunk.model_copy(deep=True)
        )

        current_roots = roots(current.text)
        duplicate = False
        for previous_roots in selected_root_sets:
            union = current_roots | previous_roots
            if union and len(current_roots & previous_roots) / len(union) >= duplicate_jaccard:
                duplicate = True
                break
        if duplicate:
            continue

        overhead = len(current.title) + 30
        remaining = max_context_chars - used_chars - overhead
        if remaining <= 180:
            break
        if len(current.text) > remaining:
            text = current.text[:remaining]
            cut = max(text.rfind(". "), text.rfind("; "), text.rfind(": "))
            if cut >= 180:
                text = text[: cut + 1]
            current = current.model_copy(deep=True, update={"text": text.strip()})
            current_roots = roots(current.text)
        selected.append(current)
        selected_root_sets.append(current_roots)
        used_chars += len(current.text) + overhead
        if len(selected) >= max_chunks:
            break
    return selected


@dataclass(slots=True)
class ExtractiveResult:
    answer: str
    used_source_ids: list[str]
    confidence: float

_ACRONYM_DEFINITIONS = {
    "соут": "специальная оценка условий труда",
    "сиз": "средства индивидуальной защиты",
}


def _definition_fallback(query: str, chunks: list[RetrievedChunk]) -> ExtractiveResult | None:
    q = " ".join(query.casefold().replace("ё", "е").split())
    if not any(marker in q for marker in ("что такое", "что означает", "расшифр")):
        return None
    for acronym, expansion in _ACRONYM_DEFINITIONS.items():
        if not re.search(rf"\b{re.escape(acronym)}\b", q):
            continue
        expected = roots(expansion)
        for chunk in chunks[:3]:
            source_roots = roots(f"{chunk.title} {chunk.text}")
            if expected and expected <= source_roots and chunk.source_id:
                return ExtractiveResult(
                    f"{acronym.upper()} — {expansion} [{chunk.source_id}].",
                    [chunk.source_id],
                    1.0,
                )
    return None


def _membership_fallback(query: str, chunks: list[RetrievedChunk]) -> ExtractiveResult | None:
    q = " ".join(query.casefold().replace("ё", "е").split())
    if not (bool(re.search(r"\bкто\s+(?:входит|состоит)\b", q)) or "состав комис" in q):
        return None
    role_markers = ("председател", "заместител", "секретар", "старший групп", "член групп")
    for chunk in chunks:
        low = chunk.text.casefold().replace("ё", "е")
        if "в состав" not in low or re.search(r"\bвход\w*\b", low) is None:
            continue
        labels: list[str] = []
        started = False
        for sentence in _sentences(chunk.text):
            s_low = sentence.casefold().replace("ё", "е")
            if "в состав" in s_low and re.search(r"\bвход\w*\b", s_low):
                started = True
                continue
            if not started:
                continue
            clean = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", sentence)
            clean = re.sub(r"\[(?:ОРГАНИЗАЦИЯ|ЧЕЛОВЕК|АДРЕС)_\d+\]", "", clean, flags=re.I)
            clean = re.sub(r"\s+", " ", clean).strip(" ;:.-")
            if not clean:
                continue
            if any(marker in clean.casefold().replace("ё", "е") for marker in role_markers):
                labels.append(clean)
            if len(labels) >= 5:
                break
        if labels and chunk.source_id:
            compact = "; ".join(labels[:5])
            return ExtractiveResult(
                f"В состав комиссии входят: {compact} [{chunk.source_id}].",
                [chunk.source_id],
                1.0,
            )
    return None


def extractive_answer(query: str, chunks: list[RetrievedChunk], *, max_sentences: int = 3) -> ExtractiveResult:
    """Deterministic, citation-safe fallback used when the local LLM is too slow."""
    structured = _definition_fallback(query, chunks) or _membership_fallback(query, chunks)
    if structured is not None:
        return structured
    qroots = roots(query)
    candidates: list[tuple[float, int, str, str, str]] = []
    for chunk_rank, chunk in enumerate(chunks):
        source_id = chunk.source_id or ""
        for sentence in _sentences(chunk.text):
            if len(sentence) < 24:
                continue
            stripped = sentence.strip()
            # Headings and list-introductions are useful context for Qwen but poor
            # standalone answers, so exclude them from the deterministic fallback.
            if (stripped.endswith(":") and len(stripped) < 220) or (
                "обязанности и действия" in stripped.casefold() and len(stripped) < 180
            ):
                continue
            score = _sentence_score(qroots, sentence, query)
            # Respect retrieval rank and its lexical signal.
            # Trust the retrieval order: the top passage has already passed
            # hybrid+lexical ranking. This suppresses superficially similar
            # sentences from lower-ranked documents in the no-LLM fallback.
            score += 0.14 - 0.16 * chunk_rank
            score += 0.12 * float(chunk.metadata.get("lexical_coverage", 0.0))
            if score > 0:
                candidates.append((score, chunk_rank, source_id, chunk.doc_id, sentence.strip()))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return ExtractiveResult("", [], 0.0)

    # When the strongest retrieved passage contains multiple concrete answer
    # sentences, keep them together instead of mixing in a definition from a
    # lower-ranked document. This is especially important for procedures/lists.
    qnorm = " ".join(query.casefold().replace("ё", "е").split())
    procedure_intent = any(
        marker in qnorm
        for marker in (
            "что делать", "как ", "порядок", "действия", "как оформ",
            "как провод", "как выполня", "какие меры", "какие требования",
        )
    )

    # In a deterministic fallback, never splice organization-specific procedures
    # from different documents. Prefer the document containing the strongest
    # answer-shaped sentence; Qwen can synthesize across documents when it finishes.
    if procedure_intent and candidates:
        best_doc = candidates[0][3]
        same_doc = [item for item in candidates if item[3] == best_doc]
        if same_doc:
            candidates = same_doc

    top_source = [item for item in candidates if item[1] == 0 and item[0] >= 0.45]
    if len(top_source) >= 2:
        top_source.sort(key=lambda item: item[0], reverse=True)
        remaining = [item for item in candidates if item[1] != 0]
        candidates = [*top_source, *remaining]

    selected: list[tuple[str, str]] = []
    seen_sentences: set[str] = set()
    used: list[str] = []
    for score, _rank, source_id, _doc_id, sentence in candidates:
        key = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        selected.append((source_id, sentence))
        if source_id and source_id not in used:
            used.append(source_id)
        if len(selected) >= max_sentences:
            break

    answer_parts = []
    for source_id, sentence in selected:
        clean = re.sub(r"\[(?:ОРГАНИЗАЦИЯ|ЧЕЛОВЕК|АДРЕС)_\d+\]", "", sentence, flags=re.I)
        clean = re.sub(r"\s+([,.;:)])", r"\1", clean)
        clean = re.sub(r"([(:])\s+", r"\1 ", clean)
        clean = re.sub(r"\s{2,}", " ", clean).strip().rstrip(" .;:")
        marker = f" [{source_id}]" if source_id else ""
        answer_parts.append(f"{clean}{marker}.")
    confidence = min(1.0, candidates[0][0])
    return ExtractiveResult(" ".join(answer_parts), used, confidence)
