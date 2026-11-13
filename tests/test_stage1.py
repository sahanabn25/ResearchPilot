"""Stage 1 regression + hardening tests. Run: python -m unittest discover -s tests -v

No network, API key or google-genai install is required; external calls are mocked.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

import agent  # noqa: E402
from researchpilot import config, llm  # noqa: E402
from researchpilot.tools import crossref  # noqa: E402
from researchpilot.tools.text_utils import clean_abstract, clean_text  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def crossref_payload():
    return {"message": {"items": [
        {"DOI": "10.1/a", "title": ["Deep <i>learning</i> for weather &amp; climate"],
         "author": [{"given": "Ada", "family": "Lovelace"}, {"family": "Turing"}],
         "published": {"date-parts": [[2024, 5]]}, "URL": "https://doi.org/10.1/a",
         "abstract": "<jats:p>Abstract This paper studies forecasting.</jats:p>",
         "container-title": ["Journal of Tests"]},
        {"DOI": "10.1/A", "title": ["Duplicate by DOI"], "published": {"date-parts": [[2020]]}},
        {"DOI": "10.1/notitle", "title": []},
        {"DOI": "10.1/b", "title": ["Second paper"], "published-print": {"date-parts": [[2021]]}},
    ]}}


class TestTextUtils(unittest.TestCase):
    def test_clean_text_strips_markup_and_entities(self):
        self.assertEqual(clean_text("A <i>b</i> &amp;  c"), "A b & c")

    def test_clean_text_none(self):
        self.assertEqual(clean_text(None), "")

    def test_clean_abstract_removes_prefix(self):
        self.assertEqual(clean_abstract("<jats:p>Abstract: We test.</jats:p>"), "We test.")


class TestCrossref(unittest.TestCase):
    def setUp(self):
        crossref.clear_cache()
        p = mock.patch.object(crossref, "_sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def test_empty_topic_rejected_without_api_call(self):
        with mock.patch.object(crossref.requests, "get") as get:
            for bad in ["", "   ", None, 5]:
                with self.assertRaises(ValueError):
                    crossref.search_papers(bad)
            get.assert_not_called()

    def test_bad_limit_rejected(self):
        for bad in [0, -1, "5", True, 10_000, 2.5]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                crossref.search_papers("topic", limit=bad)

    def test_parsing_cleaning_and_dedupe(self):
        with mock.patch.object(crossref.requests, "get", return_value=FakeResponse(payload=crossref_payload())):
            papers = crossref.search_papers("weather")
        self.assertEqual([p["doi"] for p in papers], ["10.1/a", "10.1/b"])  # dup + no-title dropped
        first = papers[0]
        self.assertEqual(first["title"], "Deep learning for weather & climate")
        self.assertEqual(first["authors"], ["Ada Lovelace", "Turing"])
        self.assertEqual(first["year"], 2024)
        self.assertEqual(first["abstract"], "This paper studies forecasting.")
        self.assertEqual(first["venue"], "Journal of Tests")
        self.assertEqual(papers[1]["year"], 2021)  # falls back to published-print
        self.assertEqual(papers[1]["abstract"], "")

    def test_cache_avoids_duplicate_calls(self):
        with mock.patch.object(crossref.requests, "get", return_value=FakeResponse(payload=crossref_payload())) as get:
            a = crossref.search_papers("Weather", limit=5)
            b = crossref.search_papers("  weather ", limit=5)
            self.assertEqual(get.call_count, 1)
        a[0]["title"] = "mutated"
        self.assertNotEqual(b[0]["title"], "mutated")  # cache returns copies

    def test_retries_503_then_succeeds(self):
        seq = [FakeResponse(503), FakeResponse(200, crossref_payload())]
        with mock.patch.object(crossref.requests, "get", side_effect=seq) as get:
            papers = crossref.search_papers("weather")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(papers), 2)

    def test_gives_up_after_max_attempts(self):
        with mock.patch.object(crossref.requests, "get", return_value=FakeResponse(503)) as get:
            with self.assertRaises(requests.HTTPError):
                crossref.search_papers("weather")
        self.assertEqual(get.call_count, config.HTTP_MAX_ATTEMPTS)

    def test_network_error_propagates_after_retries(self):
        with mock.patch.object(crossref.requests, "get", side_effect=requests.ConnectionError("down")):
            with self.assertRaises(requests.ConnectionError):
                crossref.search_papers("weather")

    def test_non_retryable_404_fails_fast(self):
        with mock.patch.object(crossref.requests, "get", return_value=FakeResponse(404)) as get:
            with self.assertRaises(requests.HTTPError):
                crossref.search_papers("weather")
        self.assertEqual(get.call_count, 1)


PAPERS = [{"title": "P1"}, {"title": "P2"}, {"title": "P3"}]


def good_plan(**overrides):
    plan = {"topic": "t", "research_questions": ["q1", "q2"], "summary": "s",
            "relevant_paper_indices": [1, 3], "limitations": "l"}
    plan.update(overrides)
    return plan


class TestValidatePlan(unittest.TestCase):
    def test_valid_plan_attaches_papers(self):
        out = agent.validate_plan(good_plan(), PAPERS)
        self.assertEqual(out["papers"], [PAPERS[0], PAPERS[2]])

    def test_invalid_indices_rejected(self):
        for bad in [0, -1, "1", True, 4, 1.5, None]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                agent.validate_plan(good_plan(relevant_paper_indices=[bad]), PAPERS)

    def test_missing_or_extra_keys_rejected(self):
        plan = good_plan()
        del plan["summary"]
        with self.assertRaises(ValueError):
            agent.validate_plan(plan, PAPERS)
        with self.assertRaises(ValueError):
            agent.validate_plan(good_plan(extra=1), PAPERS)

    def test_wrong_types_rejected(self):
        for field, bad in [("research_questions", "nope"), ("research_questions", [1]),
                           ("relevant_paper_indices", "1"), ("summary", ["x"]), ("limitations", ["x"])]:
            with self.assertRaises(ValueError, msg=field):
                agent.validate_plan(good_plan(**{field: bad}), PAPERS)

    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            agent.validate_plan([1, 2], PAPERS)


class TestLLM(unittest.TestCase):
    def setUp(self):
        llm.reset_health()
        for patcher in (mock.patch.object(llm, "_sleep", lambda s: None),
                        mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", "test-key"),
                        mock.patch.object(config, "GEMINI_FALLBACK_MODELS", [])):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def api_error(code):
        err = Exception(f"api error {code}")
        err.code = code
        return err

    def test_missing_key_raises_config_error(self):
        with mock.patch.object(config, "GEMINI_API_KEY", None):
            with self.assertRaises(llm.LLMConfigError):
                llm.generate_json("p")

    def test_success_parses_json(self):
        with mock.patch.object(llm, "_call_model", return_value=('{"a": 1}', {})):
            self.assertEqual(llm.generate_json("p"), {"a": 1})

    def test_retries_503_then_succeeds(self):
        calls = [self.api_error(503), ('{"ok": true}', {})]
        with mock.patch.object(llm, "_call_model", side_effect=calls) as m:
            self.assertEqual(llm.generate_json("p"), {"ok": True})
        self.assertEqual(m.call_count, 2)

    def test_non_transient_error_fails_fast(self):
        with mock.patch.object(llm, "list_models", return_value=[]), \
                mock.patch.object(llm, "_call_model", side_effect=self.api_error(404)) as m:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p")
        self.assertEqual(m.call_count, 1)

    def test_persistent_503_stops_at_attempt_limit(self):
        with mock.patch.object(llm, "_call_model", side_effect=self.api_error(503)) as m:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=3)
        self.assertEqual(m.call_count, 3)

    def test_invalid_json_then_valid(self):
        calls = [('{"broken":', {}), ('{"ok": 1}', {})]
        with mock.patch.object(llm, "_call_model", side_effect=calls):
            self.assertEqual(llm.generate_json("p"), {"ok": 1})

    def test_empty_response_exhausts_retries(self):
        with mock.patch.object(llm, "_call_model", return_value=("", {})) as m:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=2)
        self.assertEqual(m.call_count, 2)


class TestLLMFallback(unittest.TestCase):
    def setUp(self):
        llm.reset_health()
        for patcher in (mock.patch.object(llm, "_sleep", lambda s: None),
                        mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", "test-key"),
                        mock.patch.object(config, "GEMINI_MODEL", "primary"),
                        mock.patch.object(config, "GEMINI_FALLBACK_MODELS", ["backup"])):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def err503():
        e = Exception("503 UNAVAILABLE")
        e.code = 503
        return e

    def test_falls_back_when_primary_overloaded(self):
        def fake(prompt, system, temp, model=None):
            if model == "primary":
                raise self.err503()
            return '{"from": "backup"}', {}
        with mock.patch.object(llm, "_call_model", side_effect=fake) as m:
            self.assertEqual(llm.generate_json("p", max_attempts=3), {"from": "backup"})
        models = [c.args[3] for c in m.call_args_list]
        # "high demand" 503s are model-specific: after 2 failures we switch models rather than wait out all 3
        self.assertEqual(models, ["primary"] * 2 + ["backup"])

    def test_all_models_fail_raises_primary_error(self):
        with mock.patch.object(llm, "_call_model", side_effect=self.err503()) as m:
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p", max_attempts=3)
        self.assertIn("primary", str(ctx.exception))
        self.assertIn("backup", str(ctx.exception))
        self.assertEqual(m.call_count, 2 + 2)  # primary: switches after 2 overloaded replies; fallback capped at 2

    def test_wrong_fallback_name_does_not_mask_success_path(self):
        def fake(prompt, system, temp, model=None):
            if model == "primary":
                return '{"ok": 1}', {}
            raise AssertionError("fallback should not be used")
        with mock.patch.object(llm, "_call_model", side_effect=fake):
            self.assertEqual(llm.generate_json("p"), {"ok": 1})


class TestLLMFallbackChain(unittest.TestCase):
    def setUp(self):
        llm.reset_health()
        for patcher in (mock.patch.object(llm, "_sleep", lambda s: None),
                        mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", "test-key"),
                        mock.patch.object(config, "GEMINI_MODEL", "primary"),
                        mock.patch.object(config, "GEMINI_FALLBACK_MODELS", ["backup", "primary"])):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def err(code):
        e = Exception(f"api error {code}")
        e.code = code
        return e

    def test_chain_is_deduplicated_and_ordered(self):
        self.assertEqual(llm._model_chain(), ["primary", "backup"])

    def test_falls_back_when_primary_overloaded(self):
        def fake(prompt, system, temp, model):
            if model == "primary":
                raise self.err(503)
            return '{"from": "%s"}' % model, {}
        with mock.patch.object(llm, "_call_model", side_effect=fake) as m:
            self.assertEqual(llm.generate_json("p", max_attempts=2), {"from": "backup"})
        self.assertEqual([c.args[3] for c in m.call_args_list], ["primary", "primary", "backup"])

    def test_404_skips_straight_to_fallback(self):
        def fake(prompt, system, temp, model):
            if model == "primary":
                raise self.err(404)
            return '{"ok": 1}', {}
        with mock.patch.object(llm, "_call_model", side_effect=fake) as m:
            self.assertEqual(llm.generate_json("p"), {"ok": 1})
        self.assertEqual(m.call_count, 2)

    def test_auth_error_does_not_fall_back(self):
        with mock.patch.object(llm, "_call_model", side_effect=self.err(403)) as m:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p")
        self.assertEqual(m.call_count, 1)

    def test_all_models_overloaded_raises_after_bounded_calls(self):
        with mock.patch.object(llm, "_call_model", side_effect=self.err(503)) as m:
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p", max_attempts=2)
        self.assertEqual(m.call_count, 4)  # 2 models x 2 attempts
        self.assertIn("backup", str(ctx.exception))


class TestCreateResearchPlan(unittest.TestCase):
    def test_empty_topic_and_papers_rejected(self):
        with mock.patch.object(agent, "generate_json") as gen:
            with self.assertRaises(ValueError):
                agent.create_research_plan("  ", PAPERS)
            with self.assertRaises(ValueError):
                agent.create_research_plan("topic", [])
            gen.assert_not_called()

    def test_end_to_end_with_mocked_llm(self):
        with mock.patch.object(agent, "generate_json", return_value=good_plan()) as gen:
            out = agent.create_research_plan("topic", PAPERS)
        self.assertEqual(len(out["papers"]), 2)
        prompt = gen.call_args.args[0]
        self.assertIn("topic", prompt)
        self.assertIn(json.dumps(PAPERS[0]), prompt)

    def test_invalid_model_indices_surface_as_value_error(self):
        with mock.patch.object(agent, "generate_json", return_value=good_plan(relevant_paper_indices=[9])):
            with self.assertRaises(ValueError):
                agent.create_research_plan("topic", PAPERS)

    def test_backward_compatible_exports(self):
        for name in ("search_papers", "create_research_plan", "GEMINI_API_KEY", "MODEL", "CROSSREF_URL", "main"):
            self.assertTrue(hasattr(agent, name), name)


if __name__ == "__main__":
    unittest.main()
