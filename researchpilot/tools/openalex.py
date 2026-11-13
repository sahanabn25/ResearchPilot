"""OpenAlex tool: scholarly metadata WITH abstracts and open-access links.

Free usage only: the API key is optional and free; no payment is ever configured.
A hard per-process call ceiling (OPENALEX_MAX_CALLS) prevents runaway usage, and
quota errors surface as QuotaError so discovery can fall back to other sources.
"""
import copy

from .. import config
from ..logging_setup import get_logger
from .http import QuotaError, ServiceError, get_with_retry
from .text_utils import clean_text

log = get_logger(__name__)

_SELECT = ("id,doi,title,publication_year,authorships,abstract_inverted_index,"
           "primary_location,open_access,cited_by_count")
_CACHE: dict = {}
_calls = 0


def clear_cache() -> None:
    global _calls
    _CACHE.clear()
    _calls = 0


def reconstruct_abstract(inverted_index) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs or []:
            positions[i] = word
    return clean_text(" ".join(positions[i] for i in sorted(positions)))


def _strip_doi(value):
    if not value:
        return None
    return str(value).replace("https://doi.org/", "").replace("http://doi.org/", "").strip() or None


def _parse(item: dict):
    title = clean_text(item.get("title") or item.get("display_name"))
    if not title:
        return None
    doi = _strip_doi(item.get("doi"))
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    oa = item.get("open_access") or {}
    authors = [clean_text((a.get("author") or {}).get("display_name"))
               for a in item.get("authorships", [])]
    return {
        "title": title,
        "authors": [a for a in authors if a],
        "year": item.get("publication_year"),
        "doi": doi,
        "url": location.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else item.get("id")),
        "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
        "venue": clean_text(source.get("display_name")),
        "source": "openalex",
        "pdf_url": location.get("pdf_url") or None,
        "oa_url": oa.get("oa_url") or None,
        "open_access": bool(oa.get("is_oa")),
        "cited_by": item.get("cited_by_count"),
    }


def search_openalex(topic, limit=5):
    global _calls
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= config.MAX_PAPERS):
        raise ValueError(f"limit must be an integer between 1 and {config.MAX_PAPERS}.")

    topic = topic.strip()
    key = (topic.lower(), limit)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])

    if _calls >= config.OPENALEX_MAX_CALLS:
        raise QuotaError("OpenAlex", "local call ceiling reached (OPENALEX_MAX_CALLS)")
    if not config.OPENALEX_API_KEY:
        log.warning("OPENALEX_API_KEY not set: using the tiny keyless allowance. "
                    "Create a free key at openalex.org/settings/api.")

    params = {"search": topic, "per_page": limit, "select": _SELECT}
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY  # never logged; errors omit the URL
    _calls += 1
    response = get_with_retry("OpenAlex", config.OPENALEX_URL, params=params)

    try:
        items = response.json().get("results", [])
    except ValueError:
        raise ServiceError("OpenAlex", "malformed response") from None

    papers = [p for p in (_parse(i) for i in items) if p]
    log.info("openalex returned %d papers (calls this process: %d)", len(papers), _calls)
    _CACHE[key] = copy.deepcopy(papers)
    return papers
