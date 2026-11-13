"""Tests for LLM failure diagnostics: readable (redacted) reasons, retry hints, quota handling."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import config, llm  # noqa: E402
from researchpilot.agents.paper_analysis import analyze_papers  # noqa: E402

FAKE_KEY = "AIzaSyFAKEKEYFORTESTING1234567890abcd"
DAILY = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota. quotaId: "
         "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
PER_MINUTE = "429 RESOURCE_EXHAUSTED. Please retry in 12.5s. quotaId: GenerateRequestsPerMinutePerProjectPerModel"


def api_error(code, message):
    err = Exception(message)
    err.code = code
    return err


class TestSafeMessage(unittest.TestCase):
    def test_redacts_key_truncates_and_collapses_whitespace(self):
        with mock.patch.object(config, "GEMINI_API_KEY", "my-secret-key-123"):
            text = llm.safe_message(Exception(f"bad   key\n my-secret-key-123 and {FAKE_KEY} " + "x" * 1000), limit=120)
        self.assertNotIn("my-secret-key-123", text)
        self.assertNotIn(FAKE_KEY, text)
        self.assertEqual(len(text), 120)
        self.assertNotIn("\n", text)

    def test_retry_hint(self):
        self.assertEqual(llm.retry_hint_seconds(Exception(PER_MINUTE)), 12.5)
        self.assertEqual(llm.retry_hint_seconds(Exception("retryDelay': '30s'")), 30.0)
        self.assertEqual(llm.retry_hint_seconds(Exception("Please retry in 500s")), 60.0)   # capped
        self.assertIsNone(llm.retry_hint_seconds(Exception("no hint here")))


class TestQuotaHandling(unittest.TestCase):
    def setUp(self):
        llm.reset_health()
        for p in (mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", "test-key"),
                  mock.patch.object(config, "GEMINI_MODEL", "primary"),
                  mock.patch.object(config, "GEMINI_FALLBACK_MODELS", ["backup"])):
            p.start()
            self.addCleanup(p.stop)
        self.sleeps = []
        p = mock.patch.object(llm, "_sleep", self.sleeps.append)
        p.start()
        self.addCleanup(p.stop)

    def test_daily_quota_skips_retries_and_uses_next_model(self):
        seen = []

        def call(prompt, system, temperature, model=None):
            seen.append(model)
            if model == "primary":
                raise api_error(429, DAILY)
            return '{"ok": true}', {}
        with mock.patch.object(llm, "_call_model", side_effect=call):
            self.assertEqual(llm.generate_json("p"), {"ok": True})
        self.assertEqual(seen, ["primary", "backup"])      # one attempt on primary, not four
        self.assertEqual(self.sleeps, [])

    def test_per_minute_limit_waits_for_the_hinted_time(self):
        calls = [api_error(429, PER_MINUTE), ('{"ok": 1}', {})]
        with mock.patch.object(llm, "_call_model", side_effect=calls):
            self.assertEqual(llm.generate_json("p"), {"ok": 1})
        self.assertEqual(len(self.sleeps), 1)
        self.assertTrue(13.5 <= self.sleeps[0] <= 14.5)    # hint 12.5 + 1 cushion + <1 jitter

    def test_final_error_carries_reason_without_key(self):
        err = api_error(503, f"503 UNAVAILABLE overloaded {config.GEMINI_API_KEY}")
        with mock.patch.object(llm, "_call_model", side_effect=err):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p", max_attempts=2)
        text = str(ctx.exception)
        self.assertIn("503 UNAVAILABLE", text)
        self.assertNotIn("test-key", text)

    def test_non_transient_error_message_is_included(self):
        with mock.patch.object(llm, "_call_model", side_effect=api_error(400, "400 INVALID_ARGUMENT bad request")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p")
        self.assertIn("INVALID_ARGUMENT", str(ctx.exception))


class TestFailureReasonReachesUser(unittest.TestCase):
    def test_failed_analysis_records_reason(self):
        def failing(prompt, **kw):
            raise llm.LLMError("Gemini request failed after trying ['m'] (4 attempts each). Last error: ClientError: 429 quota")
        paper = {"paper_id": "p1", "title": "T", "year": 2020, "abstract": "x" * 200}
        out = analyze_papers([paper], llm=failing)
        stub = out["analyses"]["p1"]
        self.assertEqual(stub["status"], "failed")
        self.assertIn("429 quota", stub["error"])


class RemoteProtocolError(Exception):
    """Stands in for httpx.RemoteProtocolError (matched by class name, so httpx is not needed here)."""


class TestNetworkErrorsAndCooldown(unittest.TestCase):
    def setUp(self):
        llm.reset_health()
        for p in (mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", "test-key"),
                  mock.patch.object(config, "GEMINI_MODEL", "primary"),
                  mock.patch.object(config, "GEMINI_FALLBACK_MODELS", ["backup"]),
                  mock.patch.object(llm, "_sleep", lambda s: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(llm.reset_health)

    def test_dropped_connection_is_retried_not_fatal(self):
        calls = [RemoteProtocolError("Server disconnected without sending a response."), ('{"ok": 1}', {})]
        with mock.patch.object(llm, "_call_model", side_effect=calls) as m:
            self.assertEqual(llm.generate_json("p"), {"ok": 1})
        self.assertEqual(m.call_count, 2)

    def test_persistent_disconnects_fall_back_to_next_model(self):
        def call(prompt, system, temperature, model=None):
            if model == "primary":
                raise RemoteProtocolError("Server disconnected")
            return '{"from": "backup"}', {}
        with mock.patch.object(llm, "_call_model", side_effect=call):
            self.assertEqual(llm.generate_json("p", max_attempts=2), {"from": "backup"})

    def test_overloaded_primary_is_skipped_by_later_calls(self):
        seen = []

        def call(prompt, system, temperature, model=None):
            seen.append(model)
            if model == "primary":
                raise api_error(503, "503 UNAVAILABLE high demand")
            return '{"ok": true}', {}
        with mock.patch.object(llm, "_call_model", side_effect=call):
            llm.generate_json("first")      # pays the retries once: primary x2, then backup
            seen.clear()
            llm.generate_json("second")     # primary is cooling down: goes straight to backup
        self.assertEqual(seen, ["backup"])

    def test_cooldown_expires_and_success_clears_it(self):
        clock = [1000.0]
        with mock.patch.object(llm, "_now", lambda: clock[0]):
            llm._mark_unhealthy("primary", 90)
            self.assertEqual(llm._ordered_models(), ["backup", "primary"])   # never dropped, only deprioritised
            clock[0] += 91
            self.assertEqual(llm._ordered_models(), ["primary", "backup"])
            llm._mark_unhealthy("primary", 90)
            with mock.patch.object(llm, "_call_model", return_value=('{"ok": 1}', {})):
                llm.generate_json("p")                                       # backup answers...
            llm._unhealthy_until.pop("backup", None)
            self.assertIn("primary", llm._unhealthy_until)

    def test_real_http_and_value_errors_are_not_mistaken_for_network_errors(self):
        self.assertFalse(llm._is_transient(ValueError("bad")))
        self.assertFalse(llm._is_transient(api_error(400, "bad request")))
        self.assertTrue(llm._is_transient(RemoteProtocolError("x")))
        self.assertTrue(llm._is_transient(api_error(503, "x")))


class TestHttpRateLimitWait(unittest.TestCase):
    def test_long_retry_after_gives_up_immediately(self):
        from researchpilot.tools import http

        class Resp:
            status_code = 429
            headers = {"Retry-After": "30"}
        sleeps = []
        with mock.patch.object(http.requests, "get", return_value=Resp()) as get, \
                mock.patch.object(http, "_sleep", sleeps.append):
            with self.assertRaises(http.QuotaError):
                http.get_with_retry("OpenAlex", "https://example.org")
        self.assertEqual((get.call_count, sleeps), (1, []))      # no 30 s sleeps

    def test_short_retry_after_is_still_honoured(self):
        from researchpilot.tools import http

        class Rate:
            status_code, headers = 429, {"Retry-After": "2"}

        class Ok:
            status_code, headers = 200, {}
        sleeps = []
        with mock.patch.object(http.requests, "get", side_effect=[Rate(), Ok()]), \
                mock.patch.object(http, "_sleep", sleeps.append):
            self.assertEqual(http.get_with_retry("OpenAlex", "https://example.org").status_code, 200)
        self.assertEqual(sleeps, [2])


if __name__ == "__main__":
    unittest.main()
