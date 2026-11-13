"""Crossref tool: retrieve scholarly metadata (title, authors, year, DOI, URL, abstract).

Behaviour of the Week 2 `search_papers` is preserved; additions are input
validation, polite User-Agent, bounded retry, markup cleaning, de-duplication,
optional abstract/venue fields, and an in-process cache to avoid repeat calls.
"""
import copy
import re
import time

import requests

from .. import config
from ..logging_setup import get_logger
from .text_utils import clean_abstract, clean_text

log = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_CACHE: dict = {}
_sleep = time.sleep  # patched in tests


def clear_cache() -> None:
    _CACHE.clear()


def _headers() -> dict:
    ua = "ResearchPilot/0.3 (student research project)"
    if config.CONTACT_EMAIL:
        ua += f" (mailto:{config.CONTACT_EMAIL})"
    return {"User-Agent": ua}


def _get_with_retry(params: dict) -> requests.Response:
    attempts = config.HTTP_MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(config.CROSSREF_URL, params=params,
                                    headers=_headers(), timeout=config.HTTP_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == attempts:
                raise
            log.warning("crossref network error attempt=%d/%d: %s", attempt, attempts, type(exc).__name__)
            _sleep(min(10, 2 ** attempt))
            continue
        if response.status_code in _RETRY_STATUS and attempt < attempts:
            retry_after = str(response.headers.get("Retry-After", ""))
            delay = min(30, int(retry_after)) if retry_after.isdigit() else min(10, 2 ** attempt)
            log.warning("crossref HTTP %s attempt=%d/%d; retrying in %ss",
                        response.status_code, attempt, attempts, delay)
            _sleep(delay)
            continue
        response.raise_for_status()
        return response
    raise requests.RequestException("Crossref request failed.")  # defensive; loop always returns/raises


def _year(item: dict):
    date_info = item.get("published-print") or item.get("published-online") or item.get("published") or {}
    parts = date_info.get("date-parts", [[]])
    return parts[0][0] if parts and parts[0] else None


def _authors(item: dict) -> list:
    names = []
    for author in item.get("author", []):
        name = " ".join(p for p in [author.get("given"), author.get("family")] if p)
        if name:
            names.append(clean_text(name))
    return names


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def search_papers(topic, limit=5):
    """Tool: retrieve scholarly metadata from Crossref.

    Returns a list of dicts with keys: title, authors, year, doi, url, abstract, venue.
    `abstract` / `venue` may be empty strings (Crossref does not always provide them).
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= config.MAX_PAPERS):
        raise ValueError(f"limit must be an integer between 1 and {config.MAX_PAPERS}.")

    topic = topic.strip()
    key = (topic.lower(), limit)
    if key in _CACHE:
        log.info("crossref cache hit topic=%r limit=%d", topic, limit)
        return copy.deepcopy(_CACHE[key])

    response = _get_with_retry({
        "query.bibliographic": topic,
        "rows": limit,
        "select": "DOI,title,author,published,published-print,published-online,URL,abstract,container-title",
    })
    items = response.json().get("message", {}).get("items", [])

    papers, seen = [], set()
    for item in items:
        titles = item.get("title", [])
        if not titles:
            continue
        title = clean_text(titles[0])
        doi = item.get("DOI")
        dedupe_key = (doi or "").lower() or _norm_title(title)
        if not title or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        venues = item.get("container-title") or [""]
        papers.append({
            "title": title,
            "authors": _authors(item),
            "year": _year(item),
            "doi": doi,
            "url": item.get("URL"),
            "abstract": clean_abstract(item.get("abstract")),
            "venue": clean_text(venues[0]),
        })

    log.info("crossref returned %d papers for topic=%r", len(papers), topic)
    _CACHE[key] = copy.deepcopy(papers)
    return papers
