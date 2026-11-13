"""arXiv tool: free, no API key. Returns abstracts and open-access PDF links.

arXiv asks clients to make at most one request every ~3 seconds; this module
enforces that. Covers preprints only (strong for CS / physics / maths / stats).
"""
import copy
import re
import time
import xml.etree.ElementTree as ET

from .. import config
from ..logging_setup import get_logger
from .http import ServiceError, get_with_retry
from .text_utils import clean_text

log = get_logger(__name__)

_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_STOP = {"a", "an", "and", "the", "for", "of", "in", "on", "to", "with", "using", "based",
         "by", "via", "from", "into", "its", "their", "is", "are", "at", "as"}
MIN_INTERVAL = 3.0
_CACHE: dict = {}
_last_call = 0.0
_sleep = time.sleep  # patched in tests


def clear_cache() -> None:
    global _last_call
    _CACHE.clear()
    _last_call = 0.0


def build_query(topic: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", topic.lower()) if w not in _STOP and len(w) > 1]
    words = words[:8]
    if not words:
        return f'all:"{topic.strip()}"'
    return " AND ".join(f"all:{w}" for w in words)


def _throttle() -> None:
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0 and _last_call:
        _sleep(wait)
    _last_call = time.monotonic()


def _text(entry, path):
    node = entry.find(path, _NS)
    return clean_text(node.text) if node is not None and node.text else ""


def _parse_entry(entry):
    title = _text(entry, "a:title")
    if not title:
        return None
    abs_url = _text(entry, "a:id").replace("http://", "https://")
    pdf_url = None
    for link in entry.findall("a:link", _NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = (link.get("href") or "").replace("http://", "https://") or None
    published = _text(entry, "a:published")
    return {
        "title": title,
        "authors": [clean_text(n.text) for n in entry.findall("a:author/a:name", _NS) if n.text],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "doi": _text(entry, "x:doi") or None,
        "url": abs_url or None,
        "abstract": _text(entry, "a:summary"),
        "venue": "arXiv preprint",
        "source": "arxiv",
        "arxiv_id": abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else None,
        "pdf_url": pdf_url,
        "oa_url": pdf_url,
        "open_access": True,
        "cited_by": None,
    }


def search_arxiv(topic, limit=5):
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= config.MAX_PAPERS):
        raise ValueError(f"limit must be an integer between 1 and {config.MAX_PAPERS}.")

    key = (topic.strip().lower(), limit)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])

    _throttle()
    response = get_with_retry("arXiv", config.ARXIV_URL, params={
        "search_query": build_query(topic), "start": 0, "max_results": limit,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        raise ServiceError("arXiv", "malformed response") from None

    papers = [p for p in (_parse_entry(e) for e in root.findall("a:entry", _NS)) if p]
    log.info("arxiv returned %d papers", len(papers))
    _CACHE[key] = copy.deepcopy(papers)
    return papers
