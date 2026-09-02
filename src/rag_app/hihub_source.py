from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

import httpx

from .models import KnowledgeDocument

DEFAULT_HIHUB_SECTION_IDS = [43915, 43916, 43917, 43918, 43919, 44030]


class _TextHTMLParser(HTMLParser):
    """Минимальный HTML -> text без внешних зависимостей."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "dl", "dt",
        "dd", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
        "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
        "tr", "ul",
    }
    SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if not self._skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if not self._skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self.parts.append(data)


def _normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _html_to_text(value: str) -> str:
    parser = _TextHTMLParser()
    parser.feed(value)
    parser.close()
    return _normalize_text("".join(parser.parts))


def _editorjs_to_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return _html_to_text(value)

    parts: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            # Самые содержательные поля Editor.js проверяем первыми.
            for key in ("text", "caption", "title", "message"):
                item = node.get(key)
                if isinstance(item, str) and item.strip():
                    parts.append(_html_to_text(item))
            for key, item in node.items():
                if key not in {"text", "caption", "title", "message"}:
                    collect(item)
        elif isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, str) and node.strip():
            parts.append(_html_to_text(node))

    collect(payload)
    return _normalize_text("\n".join(part for part in parts if part))


def content_to_text(content: str | None, editor_type: str | None = None) -> str:
    if not content:
        return ""
    if (editor_type or "").lower() == "editorjs":
        return _editorjs_to_text(content)
    return _html_to_text(content)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # Стабильный fallback, чтобы hash индекса не менялся при каждом запуске.
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


class HihubKnowledgeBaseSource:
    """Источник статей HiHub через API, соответствующий переданной Postman-коллекции.

    Схема работы:
    1. POST /api/auth/logininng с email/password;
    2. Bearer access_token из ответа;
    3. GET /api/knowledgebase/section/{section_id}/articles;
    4. GET /api/knowledgebase/article/{id} для полного content.
    """

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        section_id: int,
        token_name: str = "",
        timeout_seconds: float = 30.0,
        per_page: int = 200,
        max_articles: int = 0,
        client: httpx.Client | None = None,
    ) -> None:
        if not email.strip():
            raise ValueError("HIHUB_EMAIL не заполнен.")
        if not password:
            raise ValueError("HIHUB_PASSWORD не заполнен.")
        if section_id < 0:
            raise ValueError("HIHUB_SECTION_ID не может быть отрицательным.")

        self.base_url = base_url.rstrip("/")
        self.email = email.strip()
        self.password = password
        self.section_id = section_id
        self.token_name = token_name
        self.per_page = max(1, min(int(per_page), 700))
        self.max_articles = max(0, int(max_articles))
        self._access_token: str | None = None
        self.client = client or httpx.Client(
            base_url=self.base_url + "/",
            headers={"Accept": "application/json"},
            timeout=timeout_seconds,
        )

    @property
    def source_key(self) -> str:
        return f"hihub:{self.base_url}:section:{self.section_id or 'all'}"

    def _login(self) -> str:
        body: dict[str, str] = {
            "email": self.email,
            "password": self.password,
        }
        if self.token_name:
            body["token_name"] = self.token_name
        response = self.client.post(
            "api/auth/logininng",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "HiHub не принял логин/пароль "
                f"(HTTP {response.status_code}). Проверьте HIHUB_BASE_URL, "
                "HIHUB_EMAIL и HIHUB_PASSWORD."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("HiHub login вернул не JSON.") from exc
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("HiHub login не вернул access_token.")

        # HiHub web client keeps auth state in cookies as well. Persist CSRF
        # token/cookies from login response so endpoints used by the UI work
        # the same way as Postman collection.
        self._access_token = token.strip()
        csrf = self.client.cookies.get("XSRF-TOKEN") or self.client.cookies.get("csrf")
        if csrf:
            self.client.headers.update({"X-XSRF-TOKEN": csrf})
        return self._access_token

    def _authorized_get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        token = self._access_token or self._login()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                headers = {"Authorization": f"Bearer {token}"}
                csrf = self.client.cookies.get("XSRF-TOKEN") or self.client.cookies.get("csrf")
                if csrf:
                    headers["X-XSRF-TOKEN"] = csrf
                response = self.client.get(
                    path.lstrip("/"),
                    params=params,
                    headers=headers,
                    timeout=120.0,
                )
                if response.status_code == 401 and attempt < 3:
                    token = self._login()
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
                continue
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ошибка HiHub API: GET /{path.lstrip('/')} "
                    f"вернул HTTP {response.status_code}."
                ) from exc
        raise RuntimeError(
            f"Не удалось выполнить запрос к HiHub после повторных попыток: {last_error}"
        )


    def _list_sections(self) -> list[int]:
        """Возвращает все разделы базы знаний.

        HiHub web UI gets tree/full using DPoP authorization. The backend
        integration may only have the service token, therefore keep a fallback
        list of top-level sections discovered from the UI tree.
        """
        try:
            response = self._authorized_get("api/knowledgebase/tree/full")
            payload = response.json()
        except Exception:
            return DEFAULT_HIHUB_SECTION_IDS.copy()

        items = payload.values() if isinstance(payload, dict) else payload
        result: list[int] = []
        if hasattr(items, "__iter__"):
            for item in items:
                if isinstance(item, dict) and item.get("type") == "section" and item.get("id") is not None:
                    try:
                        result.append(int(item["id"]))
                    except (TypeError, ValueError):
                        pass
        return sorted(set(result)) or DEFAULT_HIHUB_SECTION_IDS.copy()

    def _list_article_summaries(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        section_ids = [self.section_id] if self.section_id else self._list_sections()

        for section_id in section_ids:
            page = 1
            while True:
                response = self._authorized_get(
                    f"api/knowledgebase/section/{section_id}/articles",
                    params={"page": page, "per_page": self.per_page},
                )
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError("HiHub SectionArticle вернул не JSON.") from exc

                if isinstance(payload, dict):
                    items = payload.get("data", [])
                    meta = payload.get("meta") or {}
                    last_page = int(meta.get("last_page") or page)
                    next_link = (payload.get("links") or {}).get("next")
                elif isinstance(payload, list):
                    items = payload
                    last_page = page
                    next_link = None
                else:
                    raise RuntimeError("HiHub SectionArticle вернул неожиданный JSON.")

                if not isinstance(items, list):
                    raise RuntimeError("В ответе HiHub поле data должно быть массивом.")
                result.extend(item for item in items if isinstance(item, dict))

                if self.max_articles and len(result) >= self.max_articles:
                    return result[: self.max_articles]
                if page >= last_page and not next_link:
                    break
                page += 1

        return result

    def _load_article(self, summary: dict[str, Any]) -> KnowledgeDocument | None:
        article_id = summary.get("id")
        if article_id is None:
            return None
        response = self._authorized_get(f"api/knowledgebase/article/{article_id}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Статья HiHub {article_id} вернула не JSON.") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise RuntimeError(f"В ответе статьи HiHub {article_id} нет объекта data.")

        merged = {**summary, **data}
        editor_type = merged.get("editor_type")
        content = content_to_text(merged.get("content"), editor_type)
        if not content:
            content = _normalize_text(str(merged.get("preview") or ""))
        if not content:
            # Пустые служебные статьи не дают полезных чанков.
            return None

        status = str(merged.get("status") or "public")
        section_id = merged.get("section_id") or self.section_id
        updated_at = _parse_datetime(merged.get("updated_at"))
        title = str(merged.get("name") or f"Статья {article_id}").strip()
        api_url = f"{self.base_url}/knowledge_base/section/{section_id}"

        return KnowledgeDocument(
            id=str(article_id),
            version=updated_at.isoformat(),
            title=title,
            section=f"HiHub / раздел {section_id}",
            content=content,
            url=api_url,
            updated_at=updated_at,
            is_actual=status.lower() != "archive",
            metadata={
                "source": "hihub",
                "section_id": section_id,
                "status": status,
                "tags": merged.get("tags") or [],
                "preview": merged.get("preview") or "",
                "editor_type": editor_type,
                "data_type": merged.get("data_type"),
            },
        )

    def list_documents(self) -> list[KnowledgeDocument]:
        summaries = self._list_article_summaries()
        documents: list[KnowledgeDocument] = []
        for summary in summaries:
            try:
                document = self._load_article(summary)
            except Exception as exc:
                article_id = summary.get("id", "unknown")
                print(f"WARNING: skip HiHub article {article_id}: {exc}")
                continue
            if document is not None and document.is_actual:
                documents.append(document)
        return documents
