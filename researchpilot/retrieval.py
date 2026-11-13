"""Knowledge layer: chunk -> embed/index -> retrieve, plus a retrieval-only evaluation.

Two interchangeable, FREE backends:
  ChromaStore  - ChromaDB (open source) with its built-in local ONNX embedder
                 (all-MiniLM-L6-v2). Needs a one-time model download on first use.
  LexicalStore - dependency-free BM25. Used automatically if Chroma is missing or
                 its embedder cannot load (e.g. offline), so the app never breaks.

Retrieval is testable WITHOUT any LLM: see evaluate_retrieval().
"""
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "and", "the", "for", "of", "in", "on", "to", "with", "by", "is", "are",
         "was", "were", "be", "as", "at", "from", "this", "that", "it", "its", "or", "we",
         "our", "can", "which", "has", "have", "using", "used", "use"}


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> List[str]:
    """Split text into overlapping word windows (sentence-agnostic, deterministic)."""
    if not isinstance(max_words, int) or max_words < 1 or not isinstance(overlap, int) \
            or overlap < 0 or overlap >= max_words:
        raise ValueError("Need max_words >= 1 and 0 <= overlap < max_words.")
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


class LexicalStore:
    """BM25 (Okapi) over chunks. Pure Python, no downloads."""

    backend_name = "lexical-bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: Dict[str, Chunk] = {}
        self._tf: Dict[str, Counter] = {}

    def add(self, chunks: List[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c
            self._tf[c.id] = Counter(tokenize(c.text))

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._tf.clear()

    def search(self, query: str, k: int = 5, where: Optional[dict] = None) -> List[Hit]:
        if not self._chunks or k < 1:
            return []
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        n = len(self._chunks)
        lengths = {cid: sum(tf.values()) for cid, tf in self._tf.items()}
        avg_len = (sum(lengths.values()) / n) or 1.0
        df = Counter()
        for tf in self._tf.values():
            for term in q_terms:
                if term in tf:
                    df[term] += 1
        hits = []
        for cid, chunk in self._chunks.items():
            if where and any(chunk.metadata.get(key) != val for key, val in where.items()):
                continue
            tf, score = self._tf[cid], 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * lengths[cid] / avg_len))
            if score > 0:
                hits.append(Hit(cid, chunk.text, chunk.metadata, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def _clean_meta(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool."""
    out = {}
    for key, val in meta.items():
        if val is None:
            continue
        out[key] = val if isinstance(val, (str, int, float, bool)) else ", ".join(map(str, val)) \
            if isinstance(val, (list, tuple)) else str(val)
    return out


def _chroma_where(where: Optional[dict]):
    if not where:
        return None
    clauses = [{k: {"$eq": v}} for k, v in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    """ChromaDB-backed semantic store (cosine similarity, local embeddings)."""

    backend_name = "chroma"

    def __init__(self, collection: Optional[str] = None, persist_dir: Optional[str] = None):
        import chromadb  # lazy: optional dependency

        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(
            name=collection or f"rp_{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"})

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self._col.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                         metadatas=[_clean_meta(c.metadata) or {"_": ""} for c in chunks])

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        ids = self._col.get().get("ids", [])
        if ids:
            self._col.delete(ids=ids)

    def search(self, query: str, k: int = 5, where: Optional[dict] = None) -> List[Hit]:
        total = self.count()
        if total == 0 or k < 1 or not (query or "").strip():
            return []
        kwargs = {"query_texts": [query], "n_results": min(k, total)}
        w = _chroma_where(where)
        if w:
            kwargs["where"] = w
        res = self._col.query(**kwargs)
        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        return [Hit(i, d, m or {}, 1.0 - float(dist)) for i, d, m, dist in zip(ids, docs, metas, dists)]


def get_store(collection: Optional[str] = None, backend: Optional[str] = None):
    """Return a working store. 'auto' prefers Chroma but verifies it end-to-end first."""
    backend = (backend or config.VECTOR_BACKEND).lower()
    if backend == "lexical":
        return LexicalStore()
    try:
        store = ChromaStore(collection=collection, persist_dir=config.CHROMA_DIR or None)
        store.add([Chunk("__probe__", "probe text for embedder check", {"probe": True})])
        store.search("probe", k=1)  # forces the embedding model to load now, not mid-run
        store.clear()
        log.info("vector store: chroma")
        return store
    except Exception as exc:
        if backend == "chroma":
            raise
        log.warning("Chroma unavailable (%s); falling back to built-in BM25 store.", type(exc).__name__)
        return LexicalStore()


def index_papers(store, papers: List[dict], max_words: int = 180, overlap: int = 30) -> int:
    """Chunk title+abstract of each paper into the store. Returns chunks written.

    Re-indexing the same paper_id replaces its chunks (idempotent, no duplicates).
    """
    chunks = []
    for p in papers:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        pid = p.get("paper_id") or title.lower()
        body = f"{title}. {p.get('abstract', '')}".strip()
        for i, text in enumerate(chunk_text(body, max_words, overlap)):
            chunks.append(Chunk(f"{pid}#{i}", text, {
                "paper_id": pid, "title": title, "year": p.get("year") or 0,
                "has_abstract": bool(p.get("abstract")), "chunk_index": i,
                "source": ",".join(p.get("sources", [])) if p.get("sources") else "",
            }))
    store.add(chunks)
    return len(chunks)


def retrieve(store, query: str, k: int = 5, paper_id: Optional[str] = None) -> List[Hit]:
    return store.search(query, k=k, where={"paper_id": paper_id} if paper_id else None)


def evaluate_retrieval(store, cases: List[dict], k: int = 3) -> dict:
    """Retrieval-only metrics (no LLM). cases: [{"query": str, "expected": [paper_id, ...]}]."""
    if not cases:
        raise ValueError("No evaluation cases supplied.")
    hits_at_k, rr_sum, details = 0, 0.0, []
    for case in cases:
        expected = set(case["expected"])
        ranked = []
        for h in store.search(case["query"], k=k):
            pid = h.metadata.get("paper_id")
            if pid not in ranked:
                ranked.append(pid)
        rank = next((i + 1 for i, pid in enumerate(ranked) if pid in expected), None)
        hits_at_k += rank is not None
        rr_sum += 1.0 / rank if rank else 0.0
        details.append({"query": case["query"], "rank": rank})
    n = len(cases)
    return {"k": k, "recall_at_k": hits_at_k / n, "mrr": rr_sum / n, "n": n, "details": details}
