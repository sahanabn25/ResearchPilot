"""Stage 2 tests: OpenAlex, arXiv, discovery/merge, retrieval (tested WITHOUT any LLM), memory.

Run: python -m unittest discover -s tests -v
All external calls are mocked; no network, API key, chromadb or google-genai needed.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from researchpilot import config, discovery, retrieval  # noqa: E402
from researchpilot.memory import LongTermMemory, ShortTermMemory, search_key  # noqa: E402
from researchpilot.tools import arxiv, http, openalex  # noqa: E402
from researchpilot.tools.http import QuotaError, ServiceError  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code, self._payload, self.text, self.headers = status, payload, text, headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def patch_http(test, **kw):
    p1 = mock.patch.object(http.requests, "get", **kw)
    p2 = mock.patch.object(http, "_sleep", lambda s: None)
    m = p1.start()
    p2.start()
    test.addCleanup(p1.stop)
    test.addCleanup(p2.stop)
    return m


OPENALEX_PAYLOAD = {"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/ABC", "title": "Rain <b>nowcasting</b> with CNNs",
     "publication_year": 2023,
     "authorships": [{"author": {"display_name": "Ada L"}}, {"author": {"display_name": "Bo K"}}],
     "abstract_inverted_index": {"We": [0], "predict": [1], "rainfall": [2], "with": [3], "CNNs": [4]},
     "primary_location": {"landing_page_url": "https://example.org/p", "pdf_url": "https://example.org/p.pdf",
                          "source": {"display_name": "Journal X"}},
     "open_access": {"is_oa": True, "oa_url": "https://example.org/oa"}, "cited_by_count": 12},
    {"id": "https://openalex.org/W2", "doi": None, "title": "", "abstract_inverted_index": None},
]}

ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry>
  <id>http://arxiv.org/abs/2401.00001v1</id>
  <published>2024-01-02T00:00:00Z</published>
  <title>Graph networks for
   weather forecasting</title>
  <summary> We forecast weather with graph neural networks. </summary>
  <author><name>Cy D</name></author><author><name>Eve F</name></author>
  <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
  <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
  <arxiv:doi>10.9/xyz</arxiv:doi>
 </entry>
</feed>"""


class TestHttpHelper(unittest.TestCase):
    def test_quota_status_raises_quota_error(self):
        patch_http(self, return_value=FakeResponse(409))
        with self.assertRaises(QuotaError):
            http.get_with_retry("Svc", "https://x.test")

    def test_429_retries_then_quota(self):
        m = patch_http(self, return_value=FakeResponse(429))
        with self.assertRaises(QuotaError):
            http.get_with_retry("Svc", "https://x.test", max_attempts=3)
        self.assertEqual(m.call_count, 3)

    def test_error_message_never_contains_url_or_key(self):
        patch_http(self, side_effect=requests.ConnectionError("failed https://x.test?api_key=SECRET"))
        with self.assertRaises(ServiceError) as ctx:
            http.get_with_retry("Svc", "https://x.test", params={"api_key": "SECRET"})
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertNotIn("x.test", str(ctx.exception))

    def test_404_fails_fast(self):
        m = patch_http(self, return_value=FakeResponse(404))
        with self.assertRaises(ServiceError):
            http.get_with_retry("Svc", "https://x.test")
        self.assertEqual(m.call_count, 1)


class TestOpenAlex(unittest.TestCase):
    def setUp(self):
        openalex.clear_cache()

    def test_reconstruct_abstract(self):
        self.assertEqual(openalex.reconstruct_abstract({"b": [1], "a": [0], "c": [2, 3]}), "a b c c")
        self.assertEqual(openalex.reconstruct_abstract(None), "")

    def test_parse_and_key_sent(self):
        m = patch_http(self, return_value=FakeResponse(payload=OPENALEX_PAYLOAD))
        with mock.patch.object(config, "OPENALEX_API_KEY", "k123"):
            papers = openalex.search_openalex("rain nowcasting", 5)
        self.assertEqual(len(papers), 1)  # blank-title record dropped
        p = papers[0]
        self.assertEqual(p["title"], "Rain nowcasting with CNNs")
        self.assertEqual(p["doi"], "10.1/ABC")
        self.assertEqual(p["abstract"], "We predict rainfall with CNNs")
        self.assertEqual(p["authors"], ["Ada L", "Bo K"])
        self.assertTrue(p["open_access"])
        self.assertEqual(p["pdf_url"], "https://example.org/p.pdf")
        self.assertEqual(m.call_args.kwargs["params"]["api_key"], "k123")

    def test_works_without_key(self):
        m = patch_http(self, return_value=FakeResponse(payload=OPENALEX_PAYLOAD))
        with mock.patch.object(config, "OPENALEX_API_KEY", ""):
            openalex.search_openalex("topic", 5)
        self.assertNotIn("api_key", m.call_args.kwargs["params"])

    def test_cache_and_call_ceiling(self):
        m = patch_http(self, return_value=FakeResponse(payload=OPENALEX_PAYLOAD))
        with mock.patch.object(config, "OPENALEX_MAX_CALLS", 1):
            openalex.search_openalex("one", 5)
            openalex.search_openalex("one", 5)  # cached: free
            self.assertEqual(m.call_count, 1)
            with self.assertRaises(QuotaError):
                openalex.search_openalex("two", 5)
        self.assertEqual(m.call_count, 1)

    def test_quota_response_becomes_quota_error(self):
        patch_http(self, return_value=FakeResponse(409))
        with self.assertRaises(QuotaError):
            openalex.search_openalex("topic", 5)

    def test_bad_input(self):
        for bad in ["", "  ", None]:
            with self.assertRaises(ValueError):
                openalex.search_openalex(bad)
        with self.assertRaises(ValueError):
            openalex.search_openalex("t", limit=0)

    def test_malformed_json(self):
        patch_http(self, return_value=FakeResponse(payload=None))
        with self.assertRaises(ServiceError):
            openalex.search_openalex("topic", 5)


class TestArxiv(unittest.TestCase):
    def setUp(self):
        arxiv.clear_cache()

    def test_build_query_drops_stopwords(self):
        self.assertEqual(arxiv.build_query("Machine learning for weather forecasting"),
                         "all:machine AND all:learning AND all:weather AND all:forecasting")

    def test_parse_entry(self):
        patch_http(self, return_value=FakeResponse(text=ARXIV_XML))
        with mock.patch.object(arxiv, "_sleep", lambda s: None):
            papers = arxiv.search_arxiv("weather graph", 5)
        p = papers[0]
        self.assertEqual(p["title"], "Graph networks for weather forecasting")
        self.assertEqual(p["abstract"], "We forecast weather with graph neural networks.")
        self.assertEqual(p["authors"], ["Cy D", "Eve F"])
        self.assertEqual(p["year"], 2024)
        self.assertEqual(p["doi"], "10.9/xyz")
        self.assertEqual(p["arxiv_id"], "2401.00001v1")
        self.assertEqual(p["pdf_url"], "https://arxiv.org/pdf/2401.00001v1")
        self.assertTrue(p["url"].startswith("https://"))

    def test_throttle_between_calls(self):
        patch_http(self, return_value=FakeResponse(text=ARXIV_XML))
        sleeps = []
        with mock.patch.object(arxiv, "_sleep", sleeps.append):
            arxiv.search_arxiv("first topic", 5)
            self.assertEqual(sleeps, [])  # first call: no wait
            arxiv.search_arxiv("second topic", 5)
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0)

    def test_malformed_xml(self):
        patch_http(self, return_value=FakeResponse(text="<feed><oops"))
        with mock.patch.object(arxiv, "_sleep", lambda s: None):
            with self.assertRaises(ServiceError):
                arxiv.search_arxiv("topic", 5)


def raw(title, doi=None, abstract="", source="x", **extra):
    return {"title": title, "doi": doi, "abstract": abstract, "source": source, "authors": [], **extra}


class TestDiscovery(unittest.TestCase):
    def fetchers(self, **overrides):
        base = {
            "crossref": lambda t, n: [raw("Deep Learning for Rain", "10.1/A", "", "crossref"),
                                      raw("Only Crossref", "10.1/C", "", "crossref")],
            "openalex": lambda t, n: [raw("Deep learning for rain!", "10.1/a", "Long abstract about rain.", "openalex",
                                          pdf_url="https://p.pdf", cited_by=5)],
            "arxiv": lambda t, n: [raw("Graph nets", None, "Arxiv abstract", "arxiv", arxiv_id="2401.1")],
        }
        base.update(overrides)
        return base

    def test_merge_by_doi_keeps_best_fields(self):
        out = discovery.discover_papers("rain", 10, fetchers=self.fetchers())
        by_title = {p["title"].lower().rstrip("!"): p for p in out["papers"]}
        merged = by_title["deep learning for rain"]
        self.assertEqual(merged["abstract"], "Long abstract about rain.")
        self.assertEqual(merged["pdf_url"], "https://p.pdf")
        self.assertEqual(merged["cited_by"], 5)
        self.assertEqual(sorted(merged["sources"]), ["crossref", "openalex"])
        self.assertEqual(len(out["papers"]), 3)  # 4 raw records, 1 duplicate pair

    def test_merge_by_normalised_title_when_no_doi(self):
        out = discovery.discover_papers("t", 10, fetchers=self.fetchers(
            crossref=lambda t, n: [raw("Same  Title!", None, "", "crossref")],
            openalex=lambda t, n: [raw("same title", "10.5/z", "abs", "openalex")],
            arxiv=lambda t, n: []))
        self.assertEqual(len(out["papers"]), 1)
        self.assertEqual(out["papers"][0]["doi"], "10.5/z")
        self.assertTrue(out["papers"][0]["paper_id"].startswith("doi:"))

    def test_abstract_papers_ranked_first_and_limit_applied(self):
        out = discovery.discover_papers("rain", 2, fetchers=self.fetchers())
        self.assertEqual(len(out["papers"]), 2)
        self.assertTrue(all(p["abstract"] for p in out["papers"]))

    def test_one_source_failing_is_isolated(self):
        def boom(t, n):
            raise ServiceError("OpenAlex", "HTTP 500", 500)
        out = discovery.discover_papers("rain", 10, fetchers=self.fetchers(openalex=boom))
        self.assertTrue(out["source_status"]["openalex"].startswith("error"))
        self.assertTrue(out["source_status"]["crossref"].startswith("ok"))
        self.assertGreater(len(out["papers"]), 0)

    def test_quota_is_reported_as_quota(self):
        def quota(t, n):
            raise QuotaError("OpenAlex", "quota")
        out = discovery.discover_papers("rain", 10, fetchers=self.fetchers(openalex=quota))
        self.assertTrue(out["source_status"]["openalex"].startswith("quota"))

    def test_legacy_requests_error_isolated(self):
        def boom(t, n):
            raise requests.HTTPError("HTTP 500 for url https://secret")
        out = discovery.discover_papers("rain", 10, fetchers=self.fetchers(crossref=boom))
        self.assertTrue(out["source_status"]["crossref"].startswith("error"))
        self.assertNotIn("secret", out["source_status"]["crossref"])

    def test_all_sources_fail_raises(self):
        def boom(t, n):
            raise ServiceError("S", "down")
        with self.assertRaises(discovery.DiscoveryError) as ctx:
            discovery.discover_papers("rain", 10, fetchers={"crossref": boom, "openalex": boom, "arxiv": boom})
        self.assertEqual(len(ctx.exception.source_status), 3)

    def test_zero_results_is_not_an_error(self):
        empty = lambda t, n: []  # noqa: E731
        out = discovery.discover_papers("rain", 10, fetchers={"crossref": empty, "openalex": empty, "arxiv": empty})
        self.assertEqual(out["papers"], [])

    def test_cache_avoids_repeat_calls(self):
        mem = LongTermMemory(":memory:")
        calls = []
        f = {n: (lambda t, k, n=n: calls.append(n) or [raw(f"{n} paper", f"10.1/{n}", "abs", n)])
             for n in ("crossref", "openalex", "arxiv")}
        first = discovery.discover_papers("Rain", 5, memory=mem, fetchers=f)
        second = discovery.discover_papers("  rain ", 5, memory=mem, fetchers=f)
        self.assertEqual(len(calls), 3)  # second call served from memory
        self.assertTrue(second["from_cache"])
        self.assertEqual(first["papers"], second["papers"])

    def test_partial_failure_not_cached(self):
        mem = LongTermMemory(":memory:")

        def boom(t, n):
            raise ServiceError("S", "down")
        f = self.fetchers(arxiv=boom)
        discovery.discover_papers("rain", 5, memory=mem, fetchers=f)
        self.assertIsNone(mem.get_search(search_key("rain", 5, discovery.DEFAULT_SOURCES)))

    def test_validation(self):
        for args in [("", 5), ("  ", 5), ("t", 0), ("t", True), ("t", 999)]:
            with self.assertRaises(ValueError, msg=str(args)):
                discovery.discover_papers(*args, fetchers=self.fetchers())
        with self.assertRaises(ValueError):
            discovery.discover_papers("t", 5, sources=("nonsense",))


CORPUS = [
    {"paper_id": "p1", "title": "Convolutional networks for rainfall nowcasting",
     "abstract": "We apply convolutional neural networks to radar images to predict short-term rainfall.", "year": 2023},
    {"paper_id": "p2", "title": "Graph neural networks for global weather forecasting",
     "abstract": "A graph neural network models atmosphere dynamics and beats numerical weather prediction baselines.",
     "year": 2024},
    {"paper_id": "p3", "title": "Crop yield prediction from satellite imagery",
     "abstract": "Random forests and satellite vegetation indices estimate wheat and maize yield.", "year": 2022},
    {"paper_id": "p4", "title": "Transformer language models for clinical notes",
     "abstract": "Fine-tuned transformers extract diagnoses from hospital clinical text.", "year": 2021},
    {"paper_id": "p5", "title": "Federated learning privacy in mobile keyboards",
     "abstract": "Federated averaging trains next-word prediction without uploading user text.", "year": 2020},
    {"paper_id": "p6", "title": "Title only record", "abstract": "", "year": 2019},
]
EVAL_CASES = [
    {"query": "predict rainfall from radar with CNN", "expected": ["p1"]},
    {"query": "graph neural network atmosphere forecasting", "expected": ["p2"]},
    {"query": "satellite wheat yield estimation", "expected": ["p3"]},
    {"query": "clinical text diagnoses transformer", "expected": ["p4"]},
    {"query": "federated privacy keyboard", "expected": ["p5"]},
]


class TestChunking(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(retrieval.chunk_text("one two three", 10, 2), ["one two three"])

    def test_overlap_and_coverage(self):
        words = [f"w{i}" for i in range(100)]
        chunks = retrieval.chunk_text(" ".join(words), max_words=30, overlap=10)
        self.assertTrue(all(len(c.split()) <= 30 for c in chunks))
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])  # overlap preserved
        covered = set(w for c in chunks for w in c.split())
        self.assertEqual(covered, set(words))  # nothing lost

    def test_empty_and_invalid(self):
        self.assertEqual(retrieval.chunk_text("   "), [])
        for mw, ov in [(0, 0), (10, 10), (10, -1), (10, 11)]:
            with self.assertRaises(ValueError):
                retrieval.chunk_text("a b c", mw, ov)


class TestLexicalRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = retrieval.LexicalStore()
        retrieval.index_papers(self.store, CORPUS)

    def test_top1_is_relevant_paper(self):
        for case in EVAL_CASES:
            hits = retrieval.retrieve(self.store, case["query"], k=1)
            self.assertEqual(hits[0].metadata["paper_id"], case["expected"][0], case["query"])

    def test_evaluate_retrieval_metrics(self):
        report = retrieval.evaluate_retrieval(self.store, EVAL_CASES, k=3)
        self.assertEqual(report["recall_at_k"], 1.0)
        self.assertGreaterEqual(report["mrr"], 0.9)

    def test_metric_detects_failure(self):
        bad = [{"query": "satellite wheat yield", "expected": ["p5"]}]
        report = retrieval.evaluate_retrieval(self.store, bad, k=3)
        self.assertEqual(report["recall_at_k"], 0.0)
        self.assertIsNone(report["details"][0]["rank"])

    def test_filter_by_paper(self):
        hits = retrieval.retrieve(self.store, "neural networks", k=5, paper_id="p2")
        self.assertTrue(hits and all(h.metadata["paper_id"] == "p2" for h in hits))

    def test_idempotent_indexing(self):
        before = self.store.count()
        retrieval.index_papers(self.store, CORPUS)
        self.assertEqual(self.store.count(), before)

    def test_title_only_paper_indexed_and_flagged(self):
        hits = retrieval.retrieve(self.store, "title only record", k=1)
        self.assertEqual(hits[0].metadata["paper_id"], "p6")
        self.assertFalse(hits[0].metadata["has_abstract"])

    def test_edge_queries(self):
        self.assertEqual(self.store.search("", k=3), [])
        self.assertEqual(self.store.search("the of and", k=3), [])  # stopwords only
        self.assertEqual(self.store.search("zzzzqqq", k=3), [])  # no match -> nothing
        self.assertEqual(retrieval.LexicalStore().search("anything"), [])  # empty store
        self.assertEqual(self.store.search("rainfall", k=0), [])

    def test_clear(self):
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_evaluate_requires_cases(self):
        with self.assertRaises(ValueError):
            retrieval.evaluate_retrieval(self.store, [])


def make_fake_chromadb():
    class Col:
        def __init__(self):
            self.docs, self.last_query = {}, None

        def upsert(self, ids, documents, metadatas):
            for i, d, m in zip(ids, documents, metadatas):
                self.docs[i] = (d, m)

        def count(self):
            return len(self.docs)

        def get(self):
            return {"ids": list(self.docs)}

        def delete(self, ids):
            for i in ids:
                self.docs.pop(i, None)

        def query(self, query_texts, n_results, where=None):
            self.last_query = {"n": n_results, "where": where}
            q = set(retrieval.tokenize(query_texts[0]))
            scored = sorted(self.docs.items(),
                            key=lambda kv: -len(q & set(retrieval.tokenize(kv[1][0]))))[:n_results]
            return {"ids": [[i for i, _ in scored]], "documents": [[v[0] for _, v in scored]],
                    "metadatas": [[v[1] for _, v in scored]], "distances": [[0.25 for _ in scored]]}

    class Client:
        def get_or_create_collection(self, name, metadata=None):
            return Col()

    mod = types.ModuleType("chromadb")
    mod.EphemeralClient = lambda: Client()
    mod.PersistentClient = lambda path: Client()
    return mod


class TestChromaGlue(unittest.TestCase):
    """Tests OUR glue code against a fake chromadb module (not the real library)."""

    def test_add_search_scores_and_where(self):
        with mock.patch.dict(sys.modules, {"chromadb": make_fake_chromadb()}):
            store = retrieval.ChromaStore()
            retrieval.index_papers(store, CORPUS)
            hits = retrieval.retrieve(store, "rainfall radar", k=2)
            self.assertEqual(hits[0].metadata["paper_id"], "p1")
            self.assertAlmostEqual(hits[0].score, 0.75)  # 1 - distance
            retrieval.retrieve(store, "rainfall", k=2, paper_id="p1")
            self.assertEqual(store._col.last_query["where"], {"paper_id": {"$eq": "p1"}})
            retrieval.retrieve(store, "x", k=2)
            self.assertIsNone(store._col.last_query["where"])

    def test_metadata_cleaning(self):
        self.assertEqual(retrieval._clean_meta({"a": None, "b": [1, 2], "c": 3, "d": True}),
                         {"b": "1, 2", "c": 3, "d": True})
        self.assertEqual(retrieval._chroma_where({"a": 1, "b": 2}),
                         {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]})

    def test_auto_uses_chroma_when_healthy(self):
        with mock.patch.dict(sys.modules, {"chromadb": make_fake_chromadb()}):
            store = retrieval.get_store(backend="auto")
            self.assertEqual(store.backend_name, "chroma")
            self.assertEqual(store.count(), 0)  # probe cleaned up

    def test_auto_falls_back_when_chroma_missing(self):
        with mock.patch.dict(sys.modules, {"chromadb": None}):
            self.assertEqual(retrieval.get_store(backend="auto").backend_name, "lexical-bm25")
            with self.assertRaises(Exception):
                retrieval.get_store(backend="chroma")

    def test_auto_falls_back_when_embedder_fails(self):
        mod = make_fake_chromadb()

        class BrokenClient:
            def get_or_create_collection(self, name, metadata=None):
                col = mock.Mock()
                col.upsert.side_effect = RuntimeError("model download failed")
                return col
        mod.EphemeralClient = lambda: BrokenClient()
        with mock.patch.dict(sys.modules, {"chromadb": mod}):
            self.assertEqual(retrieval.get_store(backend="auto").backend_name, "lexical-bm25")

    def test_lexical_forced(self):
        self.assertEqual(retrieval.get_store(backend="lexical").backend_name, "lexical-bm25")


class TestShortTermMemory(unittest.TestCase):
    def test_bounded_turns(self):
        m = ShortTermMemory(max_turns=3)
        for i in range(5):
            m.add_turn("user", f"msg{i}")
        self.assertEqual([t["content"] for t in m.turns], ["msg2", "msg3", "msg4"])

    def test_validation(self):
        m = ShortTermMemory()
        with self.assertRaises(ValueError):
            m.add_turn("robot", "hi")
        with self.assertRaises(ValueError):
            m.add_turn("user", 5)
        with self.assertRaises(ValueError):
            ShortTermMemory(0)

    def test_long_content_truncated_and_summary_bounded(self):
        m = ShortTermMemory()
        m.add_turn("user", "x" * 10000)
        self.assertEqual(len(m.turns[0]["content"]), 4000)
        m.set("topic", "weather")
        m.set("papers", [1, 2, 3])
        s = m.context_summary(max_chars=100)
        self.assertLessEqual(len(s), 100)
        self.assertIn("topic: weather", m.context_summary())
        self.assertIn("papers: 3 item(s)", m.context_summary())

    def test_roundtrip(self):
        m = ShortTermMemory(5)
        m.add_turn("assistant", "hello")
        m.set("k", {"a": 1})
        m2 = ShortTermMemory.from_dict(m.to_dict())
        self.assertEqual(m2.turns, m.turns)
        self.assertEqual(m2.get("k"), {"a": 1})
        self.assertIsNone(m2.get("missing"))


class TestLongTermMemory(unittest.TestCase):
    def test_search_cache_roundtrip_and_expiry(self):
        mem = LongTermMemory(":memory:")
        mem.save_search("k", "topic", [{"title": "T"}], {"crossref": "ok (1)"})
        self.assertEqual(mem.get_search("k")["papers"], [{"title": "T"}])
        self.assertIsNone(mem.get_search("missing"))
        self.assertIsNotNone(mem.get_search("k", max_age_days=7))
        self.assertIsNone(mem.get_search("k", max_age_days=-1))  # always expired

    def test_runs_and_prefs(self):
        mem = LongTermMemory(":memory:")
        mem.save_run("topic", "A", {"papers": 5})
        mem.save_run("upload", "B")
        runs = mem.recent_runs(5)
        self.assertEqual([r["topic"] for r in runs], ["B", "A"])
        self.assertEqual(runs[1]["summary"], {"papers": 5})
        with self.assertRaises(ValueError):
            mem.save_run("bogus", "C")
        mem.set_pref("paper_limit", 10)
        self.assertEqual(mem.get_pref("paper_limit"), 10)
        self.assertEqual(mem.get_pref("nope", "dflt"), "dflt")

    def test_persists_to_disk_across_instances(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "mem.sqlite3")
            LongTermMemory(path).save_run("topic", "persisted")
            self.assertEqual(LongTermMemory(path).recent_runs(1)[0]["topic"], "persisted")

    def test_sql_injection_safe(self):
        mem = LongTermMemory(":memory:")
        evil = "x'); DROP TABLE runs;--"
        mem.save_run("topic", evil)
        self.assertEqual(mem.recent_runs(1)[0]["topic"], evil)

    def test_search_key_normalises(self):
        self.assertEqual(search_key("  Rain   Fall ", 5, ["b", "a"]), search_key("rain fall", 5, ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
