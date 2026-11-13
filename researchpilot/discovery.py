"""Discovery service: query Crossref + OpenAlex + arXiv, merge duplicates, cache results.

Design points
- One source failing never aborts discovery (status is reported per source).
- Duplicates (same DOI, arXiv id or normalised title) are merged; the merged
  record keeps the longest abstract and the union of sources / links.
- Results are cached in long-term memory so repeat topics make no API calls.
- Only if EVERY source fails (and nothing is cached) is DiscoveryError raised.
"""
import hashlib
import re
from typing import Callable, Dict, Optional, Sequence

from . import config
from .logging_setup import get_logger
from .memory import LongTermMemory, search_key
from .tools.arxiv import search_arxiv
from .tools.crossref import search_papers as search_crossref
from .tools.http import QuotaError, ServiceError
from .tools.openalex import search_openalex

log = get_logger(__name__)

DEFAULT_SOURCES = ("crossref", "openalex", "arxiv")
_FETCHERS: Dict[str, Callable] = {
    "crossref": search_crossref,
    "openalex": search_openalex,
    "arxiv": search_arxiv,
}


class DiscoveryError(Exception):
    """Every source failed and nothing was cached."""

    def __init__(self, message: str, source_status: Optional[dict] = None):
        super().__init__(message)
        self.source_status = source_status or {}


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def make_paper_id(paper: dict) -> str:
    if paper.get("doi"):
        return "doi:" + paper["doi"].lower()
    if paper.get("arxiv_id"):
        return "arxiv:" + paper["arxiv_id"].lower()
    return "title:" + hashlib.sha1(_norm_title(paper.get("title", "")).encode()).hexdigest()[:12]


def to_schema(paper: dict, source: str) -> dict:
    """Normalise any source's record to the common schema."""
    out = {
        "title": paper.get("title", ""),
        "authors": list(paper.get("authors") or []),
        "year": paper.get("year"),
        "doi": (paper.get("doi") or None) and paper["doi"].lower(),
        "url": paper.get("url"),
        "abstract": paper.get("abstract") or "",
        "venue": paper.get("venue") or "",
        "arxiv_id": paper.get("arxiv_id"),
        "pdf_url": paper.get("pdf_url"),
        "oa_url": paper.get("oa_url"),
        "open_access": bool(paper.get("open_access")),
        "cited_by": paper.get("cited_by"),
        "sources": [paper.get("source") or source],
    }
    out["paper_id"] = make_paper_id(out)
    return out


def _merge_into(base: dict, new: dict) -> None:
    for field in ("title", "year", "doi", "url", "venue", "arxiv_id", "pdf_url", "oa_url"):
        if not base.get(field) and new.get(field):
            base[field] = new[field]
    if len(new.get("abstract", "")) > len(base.get("abstract", "")):
        base["abstract"] = new["abstract"]
    if len(new.get("authors", [])) > len(base.get("authors", [])):
        base["authors"] = new["authors"]
    base["open_access"] = base.get("open_access") or new.get("open_access", False)
    cites = [c for c in (base.get("cited_by"), new.get("cited_by")) if isinstance(c, int)]
    base["cited_by"] = max(cites) if cites else None
    for s in new.get("sources", []):
        if s not in base["sources"]:
            base["sources"].append(s)
    base["paper_id"] = make_paper_id(base)


def merge_papers(per_source: Sequence[Sequence[dict]]) -> list:
    """Round-robin interleave sources, merging duplicates by DOI / arXiv id / title."""
    merged, by_key = [], {}
    depth = max((len(lst) for lst in per_source), default=0)
    for i in range(depth):
        for lst in per_source:
            if i >= len(lst):
                continue
            paper = lst[i]
            keys = [k for k in (
                "doi:" + paper["doi"] if paper.get("doi") else None,
                "arxiv:" + paper["arxiv_id"].lower() if paper.get("arxiv_id") else None,
                "title:" + _norm_title(paper["title"]),
            ) if k]
            hit = next((by_key[k] for k in keys if k in by_key), None)
            if hit is not None:
                _merge_into(hit, paper)
            else:
                hit = paper
                merged.append(hit)
            for k in keys:
                by_key.setdefault(k, hit)
            if hit.get("doi"):
                by_key.setdefault("doi:" + hit["doi"], hit)
    return merged


def discover_papers(topic: str, limit: int = 10, sources: Sequence[str] = DEFAULT_SOURCES,
                    memory: Optional[LongTermMemory] = None,
                    fetchers: Optional[Dict[str, Callable]] = None) -> dict:
    """Return {"topic", "papers", "source_status", "from_cache"}."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= config.MAX_PAPERS):
        raise ValueError(f"limit must be an integer between 1 and {config.MAX_PAPERS}.")
    unknown = [s for s in sources if s not in _FETCHERS and not (fetchers and s in fetchers)]
    if unknown or not sources:
        raise ValueError(f"Unknown or empty sources: {unknown or 'none given'}")

    topic = topic.strip()
    key = search_key(topic, limit, sources)
    if memory is not None:
        cached = memory.get_search(key, max_age_days=config.SEARCH_CACHE_DAYS)
        if cached:
            log.info("discovery cache hit topic=%r", topic)
            return {"topic": topic, "papers": cached["papers"],
                    "source_status": cached["source_status"], "from_cache": True}

    table = {**_FETCHERS, **(fetchers or {})}
    per_source, status = [], {}
    for name in sources:
        try:
            raw = table[name](topic, limit)
            per_source.append([to_schema(p, name) for p in raw])
            status[name] = f"ok ({len(raw)})"
        except QuotaError as exc:
            status[name] = f"quota: {exc}"
            log.warning("discovery source %s hit quota", name)
        except (ServiceError, ValueError) as exc:
            # Our own exceptions: messages are written by us and never contain URLs/keys.
            status[name] = f"error: {type(exc).__name__}: {exc}"
            log.warning("discovery source %s failed: %s", name, type(exc).__name__)
        except Exception as exc:
            # Third-party exceptions (e.g. requests.HTTPError) can embed the full URL,
            # so only the type name is exposed.
            status[name] = f"error: {type(exc).__name__}"
            log.warning("discovery source %s failed: %s", name, type(exc).__name__)

    if not per_source:
        raise DiscoveryError("All paper sources failed: " + "; ".join(f"{k}: {v}" for k, v in status.items()),
                             status)

    papers = merge_papers(per_source)
    papers.sort(key=lambda p: not p["abstract"])  # stable: abstract-bearing papers first
    papers = papers[:limit]

    all_ok = all(v.startswith("ok") for v in status.values())
    if memory is not None and papers and all_ok:
        memory.save_search(key, topic, papers, status)  # don't cache partial results
    return {"topic": topic, "papers": papers, "source_status": status, "from_cache": False}
