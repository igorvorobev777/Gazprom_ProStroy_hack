from __future__ import annotations

from math import isfinite
from typing import Protocol, Sequence

from .schemas import RetrievedChunk


class PassageScorer(Protocol):
    """Минимальный контракт модели, оценивающей пары [query, passage]."""

    def predict(
        self,
        pairs: list[list[str]],
        *,
        batch_size: int,
    ) -> Sequence[float]: ...


class PassageSelector(Protocol):
    def select(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]: ...


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _build_windows(
    text: str,
    *,
    window_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int, str]]:
    """Создаёт перекрывающиеся окна и старается не резать крайнее слово."""

    normalized = _normalize_text(text)

    if not normalized:
        return []

    if len(normalized) <= window_chars:
        return [(0, len(normalized), normalized)]

    step = window_chars - overlap_chars
    windows: list[tuple[int, int, str]] = []

    raw_start = 0

    while raw_start < len(normalized):
        raw_end = min(
            raw_start + window_chars,
            len(normalized),
        )

        start = raw_start
        end = raw_end

        # Не начинать окно с середины слова.
        if start > 0:
            next_space = normalized.find(
                " ",
                start,
                min(start + 40, len(normalized)),
            )

            if next_space != -1:
                start = next_space + 1

        # Не заканчивать окно серединой слова.
        if end < len(normalized):
            previous_space = normalized.rfind(
                " ",
                max(start, end - 40),
                end,
            )

            if previous_space > start:
                end = previous_space

        passage = normalized[start:end].strip()

        if passage:
            windows.append(
                (
                    start,
                    end,
                    passage,
                )
            )

        if raw_end >= len(normalized):
            break

        raw_start += step

    return windows


class CrossEncoderPassageSelector:
    """Выбирает одно наиболее релевантное окно из каждого top-k чанка."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        model_max_length: int,
        window_chars: int,
        overlap_chars: int,
        batch_size: int,
        scorer: PassageScorer | None = None,
    ) -> None:
        if window_chars < 1:
            raise ValueError(
                "window_chars должен быть положительным."
            )

        if overlap_chars < 0 or overlap_chars >= window_chars:
            raise ValueError(
                "overlap_chars должен быть неотрицательным "
                "и меньше window_chars."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size должен быть положительным."
            )

        # В unit-тестах передаётся фальшивый scorer,
        # поэтому настоящая тяжёлая модель не загружается.
        if scorer is None:
            try:
                from sentence_transformers.cross_encoder import (
                    CrossEncoder,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Для passage selection установи "
                    "sentence-transformers."
                ) from exc

            scorer = CrossEncoder(
                model_name,
                max_length=model_max_length,
                device=device,
            )

        self._scorer = scorer
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size

    def select(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Вопрос для passage selection не может быть пустым."
            )

        if not chunks:
            return []

        # chunk_index, start, end, passage
        candidates: list[
            tuple[int, int, int, str]
        ] = []

        for chunk_index, chunk in enumerate(chunks):
            windows = _build_windows(
                chunk.text,
                window_chars=self.window_chars,
                overlap_chars=self.overlap_chars,
            )

            for start, end, passage in windows:
                candidates.append(
                    (
                        chunk_index,
                        start,
                        end,
                        passage,
                    )
                )

        if not candidates:
            return []

        # Все окна оцениваются одним batch.
        pairs = [
            [query, passage]
            for _, _, _, passage in candidates
        ]

        try:
            raw_scores = self._scorer.predict(
                pairs,
                batch_size=self.batch_size,
            )
        except Exception as exc:
            raise ValueError(
                "Не удалось оценить окна passage scorer."
            ) from exc

        scores = [
            float(score)
            for score in raw_scores
        ]

        if len(scores) != len(candidates):
            raise ValueError(
                "Passage scorer вернул количество оценок, "
                "не совпадающее с количеством окон."
            )

        if any(
            not isfinite(score)
            for score in scores
        ):
            raise ValueError(
                "Passage scorer вернул NaN или Infinity."
            )

        # Для каждого исходного чанка выбираем одно лучшее окно.
        best_by_chunk: dict[
            int,
            tuple[float, int, int, str],
        ] = {}

        for candidate, score in zip(
            candidates,
            scores,
            strict=True,
        ):
            chunk_index, start, end, passage = candidate

            previous = best_by_chunk.get(chunk_index)

            if previous is None or score > previous[0]:
                best_by_chunk[chunk_index] = (
                    score,
                    start,
                    end,
                    passage,
                )

        selected: list[RetrievedChunk] = []

        # Порядок top-5 после reranker сохраняется.
        for chunk_index, chunk in enumerate(chunks):
            best = best_by_chunk.get(chunk_index)

            if best is None:
                continue

            score, start, end, passage = best

            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "passage_selected": True,
                    "passage_start": start,
                    "passage_end": end,
                    "passage_score": score,
                    "original_text_chars": len(
                        _normalize_text(chunk.text)
                    ),
                    "selected_text_chars": len(passage),
                }
            )

            selected.append(
                chunk.model_copy(
                    deep=True,
                    update={
                        "text": passage,
                        "metadata": metadata,
                    },
                )
            )

        return selected