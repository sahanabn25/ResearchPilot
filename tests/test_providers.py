"""Tests for the provider layer: settings resolution, OpenAI-compatible calls (Groq etc.), error handling.

All HTTP is mocked. Run: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from researchpilot import config, llm  # noqa: E402

KEY = "gsk_TESTKEYVALUE1234567890abcdefgh"


class Resp:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code, self._payload, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def chat(content, usage=None):
    return Resp(200, {"choices": [{"message": {"content": content}}],
                      "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7}})


def err(status, message, headers=None):
    return Resp(status, {"error": {"message": message}}, headers)


class TestResolveSettings(unittest.TestCase):
    def test_groq_defaults(self):
        s = config.resolve_llm_settings("groq", {"GROQ_API_KEY": " abc "})
        self.assertEqual((s["base_url"], s["api_key"], s["model"]),
                         ("https://api.groq.com/openai/v1", "abc", "llama-3.3-70b-versatile"))
        self.assertEqual(s["fallbacks"], ["llama-3.1-8b-instant"])

    def test_env_overrides_win(self):
        s = config.resolve_llm_settings("groq", {"GROQ_API_KEY": "a", "LLM_API_KEY": "b", "LLM_MODEL": "m1",
                                                 "LLM_BASE_URL": "http://x/v1", "LLM_FALLBACK_MODELS": "m2, m3 ,"})
        self.assertEqual((s["api_key"], s["model"], s["base_url"], s["fallbacks"]), ("b", "m1", "http://x/v1", ["m2", "m3"]))

    def test_blank_fallbacks_turn_fallback_off(self):
        self.assertEqual(config.resolve_llm_settings("groq", {"LLM_FALLBACK_MODELS": ""})["fallbacks"], [])

    def test_other_presets(self):
        self.assertEqual(config.resolve_llm_settings("openrouter", {"OPENROUTER_API_KEY": "k"})["base_url"],
                         "https://openrouter.ai/api/v1")
        ollama = config.resolve_llm_settings("ollama", {})
        self.assertEqual((ollama["api_key"], ollama["base_url"]), ("", "http://localhost:11434/v1"))
        custom = config.resolve_llm_settings("openai", {"LLM_BASE_URL": "http://h/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m"})
        self.assertEqual((custom["base_url"], custom["model"]), ("http://h/v1", "m"))

    def test_unknown_provider_yields_empty_settings(self):
        s = config.resolve_llm_settings("nonsense", {})
        self.assertEqual((s["base_url"], s["api_key"], s["model"]), ("", "", ""))


class ProviderTestCase(unittest.TestCase):
    PROVIDER = "groq"

    def setUp(self):
        llm.reset_health()
        self.sleeps = []
        for p in (mock.patch.object(config, "LLM_PROVIDER", self.PROVIDER),
                  mock.patch.object(config, "LLM_API_KEY", KEY),
                  mock.patch.object(config, "LLM_BASE_URL", "https://api.example.test/openai/v1/"),
                  mock.patch.object(config, "LLM_MODEL", "m-main"),
                  mock.patch.object(config, "LLM_FALLBACK_MODELS", ["m-fallback"]),
                  mock.patch.object(llm, "_sleep", self.sleeps.append)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(llm.reset_health)


class TestConfigHelpers(ProviderTestCase):
    def test_is_configured_per_provider(self):
        self.assertTrue(llm.is_configured())
        with mock.patch.object(config, "LLM_API_KEY", ""):
            self.assertFalse(llm.is_configured())
            with mock.patch.object(config, "LLM_PROVIDER", "ollama"):
                self.assertTrue(llm.is_configured())                      # no key needed locally
        with mock.patch.object(config, "LLM_PROVIDER", "nonsense"):
            self.assertFalse(llm.is_configured())
            self.assertIn("Unknown LLM_PROVIDER", llm.missing_credentials_message())
        with mock.patch.object(config, "LLM_PROVIDER", "gemini"), mock.patch.object(config, "GEMINI_API_KEY", None):
            self.assertFalse(llm.is_configured())
            self.assertIn("GEMINI_API_KEY", llm.missing_credentials_message())

    def test_missing_key_message_names_the_right_variable(self):
        self.assertIn("GROQ_API_KEY", llm.missing_credentials_message())
        with mock.patch.object(config, "LLM_PROVIDER", "openrouter"):
            self.assertIn("OPENROUTER_API_KEY", llm.missing_credentials_message())

    def test_model_chain_and_describe(self):
        self.assertEqual(llm._model_chain(), ["m-main", "m-fallback"])
        with mock.patch.object(config, "LLM_FALLBACK_MODELS", ["m-main", "x"]):
            self.assertEqual(llm._model_chain(), ["m-main", "x"])         # no duplicates
        self.assertEqual(llm.describe(), "groq · m-main")

    def test_unconfigured_generate_json_raises_config_error(self):
        with mock.patch.object(config, "LLM_API_KEY", ""):
            with self.assertRaises(llm.LLMConfigError):
                llm.generate_json("p")


class TestOpenAICompatCall(ProviderTestCase):
    def test_request_shape_and_parsing(self):
        with mock.patch.object(llm.requests, "post", return_value=chat('{"ok": true}')) as post:
            data = llm.generate_json("USER PROMPT", system_instruction="Return JSON only.", temperature=0.3)
        self.assertEqual(data, {"ok": True})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.example.test/openai/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {KEY}")
        body = kwargs["json"]
        self.assertEqual((body["model"], body["temperature"]), ("m-main", 0.3))
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["messages"][1]["content"], "USER PROMPT")
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_no_authorization_header_without_a_key(self):
        with mock.patch.object(config, "LLM_PROVIDER", "ollama"), mock.patch.object(config, "LLM_API_KEY", ""):
            with mock.patch.object(llm.requests, "post", return_value=chat("{}")) as post:
                llm.generate_json("p")
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_json_mode_rejected_retries_without_response_format(self):
        seq = [err(400, "response_format is not supported by this model"), chat('{"ok": 1}')]
        with mock.patch.object(llm.requests, "post", side_effect=seq) as post:
            self.assertEqual(llm.generate_json("p"), {"ok": 1})
        self.assertIn("response_format", post.call_args_list[0].kwargs["json"])
        self.assertNotIn("response_format", post.call_args_list[1].kwargs["json"])

    def test_tolerates_fenced_or_chatty_json(self):
        for content in ('```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} Hope that helps.',
                        '<think>hmm</think>{"a": 1}'):
            with mock.patch.object(llm.requests, "post", return_value=chat(content)):
                self.assertEqual(llm.generate_json("p"), {"a": 1}, content)

    def test_garbage_output_retries_then_fails_clearly(self):
        with mock.patch.object(llm.requests, "post", return_value=chat("not json at all")) as post:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=2)
        self.assertEqual(post.call_count, 2 + 2)                        # primary 2 + fallback 2

    def test_unexpected_response_shape_is_a_bad_output(self):
        with mock.patch.object(llm.requests, "post", return_value=Resp(200, {"nope": 1})) as post:
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=1)
        self.assertGreaterEqual(post.call_count, 1)


class TestProviderErrors(ProviderTestCase):
    def test_rate_limit_waits_for_groq_hint_then_succeeds(self):
        seq = [err(429, "Rate limit reached for model on tokens per minute (TPM). Please try again in 2.5s."),
               chat('{"ok": 1}')]
        with mock.patch.object(llm.requests, "post", side_effect=seq):
            self.assertEqual(llm.generate_json("p"), {"ok": 1})
        self.assertEqual(len(self.sleeps), 1)
        self.assertTrue(3.5 <= self.sleeps[0] <= 4.5)                   # 2.5 hint + 1 cushion + <1 jitter

    def test_retry_after_header_is_used(self):
        seq = [err(429, "slow down", headers={"Retry-After": "7"}), chat('{"ok": 1}')]
        with mock.patch.object(llm.requests, "post", side_effect=seq):
            llm.generate_json("p")
        self.assertTrue(8.0 <= self.sleeps[0] <= 9.0)

    def test_daily_token_quota_switches_model_without_retrying(self):
        calls = []

        def post(url, headers=None, json=None, timeout=None):
            calls.append(json["model"])
            if json["model"] == "m-main":
                return err(429, "Rate limit reached ... on tokens per day (TPD): Limit 100000, Used 99990")
            return chat('{"from": "fallback"}')
        with mock.patch.object(llm.requests, "post", side_effect=post):
            self.assertEqual(llm.generate_json("p"), {"from": "fallback"})
        self.assertEqual(calls, ["m-main", "m-fallback"])
        self.assertEqual(self.sleeps, [])

    def test_bad_key_fails_fast_with_readable_message_and_no_key_leak(self):
        with mock.patch.object(llm.requests, "post", return_value=err(401, f"Invalid API key {KEY}")) as post:
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p")
        self.assertEqual(post.call_count, 1)
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))

    def test_model_not_found_moves_to_next_model(self):
        def post(url, headers=None, json=None, timeout=None):
            return err(404, "model not found") if json["model"] == "m-main" else chat('{"ok": 2}')
        with mock.patch.object(llm.requests, "post", side_effect=post):
            self.assertEqual(llm.generate_json("p"), {"ok": 2})

    def test_connection_errors_are_retried(self):
        seq = [requests.ConnectionError("reset"), chat('{"ok": 3}')]
        with mock.patch.object(llm.requests, "post", side_effect=seq):
            self.assertEqual(llm.generate_json("p"), {"ok": 3})

    def test_server_errors_are_retried_then_succeed(self):
        seq = [err(500, "internal"), chat('{"ok": 4}')]
        with mock.patch.object(llm.requests, "post", side_effect=seq):
            self.assertEqual(llm.generate_json("p"), {"ok": 4})

    def test_oversized_request_fails_clearly_without_hammering(self):
        with mock.patch.object(llm.requests, "post",
                               return_value=err(413, "Request too large for model on tokens per minute")) as post:
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p")
        self.assertEqual(post.call_count, 1)
        self.assertIn("too large", str(ctx.exception))


class TestHintsAndRedaction(unittest.TestCase):
    def test_groq_style_durations(self):
        hint = llm.retry_hint_seconds
        self.assertEqual(hint(Exception("Please try again in 7.2s.")), 7.2)
        self.assertEqual(hint(Exception("Please try again in 1m3.6s")), 60.0)   # 63.6s is capped at 60
        self.assertEqual(hint(Exception("Please try again in 250ms")), 0.25)
        self.assertEqual(hint(Exception("Please try again in 1m30s")), 60.0)
        self.assertEqual(hint(Exception("HTTP 429: busy. Retry-After: 12s.")), 12.0)
        self.assertIsNone(hint(Exception("nothing useful")))

    def test_redacts_llm_key_and_key_shapes(self):
        with mock.patch.object(config, "LLM_API_KEY", "custom-secret-value"):
            text = llm.safe_message(Exception("bad custom-secret-value and gsk_ABCDEFGHIJKLMNOPQRSTUVWX and "
                                              "sk-or-v1-abcdefghijklmnopqrstuvwxyz"))
        for secret in ("custom-secret-value", "gsk_ABCDEFGHIJKLMNOPQRSTUVWX", "sk-or-v1-abcdefghijklmnopqrstuvwxyz"):
            self.assertNotIn(secret, text)


class TestListModels(ProviderTestCase):
    def test_lists_openai_compatible_models(self):
        payload = {"data": [{"id": "b-model"}, {"id": "a-model"}, {"nope": 1}]}
        with mock.patch.object(llm.requests, "get", return_value=Resp(200, payload)) as get:
            self.assertEqual(llm.list_models(), ["a-model", "b-model"])
        self.assertEqual(get.call_args.args[0], "https://api.example.test/openai/v1/models")

    def test_list_models_error_is_readable(self):
        with mock.patch.object(llm.requests, "get", return_value=err(401, "Invalid API key")):
            with self.assertRaises(llm.ProviderHTTPError) as ctx:
                llm.list_models()
        self.assertIn("401", str(ctx.exception))


class TestAutoModelDiscovery(ProviderTestCase):
    """The failure seen on a real account: every configured model returns 404."""

    AVAILABLE = ["whisper-large-v3-turbo", "meta-llama/llama-guard-4-12b", "playai-tts", "zeta-chat-model",
                 "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

    @staticmethod
    def server(good, calls):
        def post(url, headers=None, json=None, timeout=None):
            calls.append(json["model"])
            if json["model"] in good:
                return chat('{"ok": true}')
            return err(404, f"The model `{json['model']}` does not exist or you do not have access to it.")
        return post

    def test_discovers_a_working_model_when_all_configured_models_404(self):
        calls = []
        with mock.patch.object(llm, "list_models", return_value=self.AVAILABLE) as lm, \
                mock.patch.object(llm.requests, "post", side_effect=self.server({"openai/gpt-oss-120b"}, calls)):
            self.assertEqual(llm.generate_json("p"), {"ok": True})
        self.assertEqual(calls, ["m-main", "m-fallback", "openai/gpt-oss-120b"])
        self.assertEqual(lm.call_count, 1)

    def test_later_calls_go_straight_to_the_discovered_model(self):
        calls = []
        with mock.patch.object(llm, "list_models", return_value=self.AVAILABLE) as lm, \
                mock.patch.object(llm.requests, "post", side_effect=self.server({"openai/gpt-oss-120b"}, calls)):
            llm.generate_json("first")
            calls.clear()
            llm.generate_json("second")
        self.assertEqual(calls, ["openai/gpt-oss-120b"])               # no repeated 404s, no second listing
        self.assertEqual(lm.call_count, 1)

    def test_clear_error_when_nothing_usable_is_found(self):
        for listing in ([], ["whisper-large-v3", "playai-tts"]):
            llm.reset_health()
            with mock.patch.object(llm, "list_models", return_value=listing), \
                    mock.patch.object(llm.requests, "post", side_effect=self.server(set(), [])):
                with self.assertRaises(llm.LLMError) as ctx:
                    llm.generate_json("p")
            self.assertIn("check_llm.py", str(ctx.exception))
            self.assertIn("m-main", str(ctx.exception))

    def test_listing_failure_is_handled(self):
        with mock.patch.object(llm, "list_models", side_effect=llm.ProviderHTTPError(401, "bad key")), \
                mock.patch.object(llm.requests, "post", side_effect=self.server(set(), [])):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate_json("p")
        self.assertIn("check_llm.py", str(ctx.exception))

    def test_discovered_models_that_also_fail_do_not_loop(self):
        calls = []
        with mock.patch.object(llm, "list_models", return_value=self.AVAILABLE) as lm, \
                mock.patch.object(llm.requests, "post", side_effect=self.server(set(), calls)):
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p")
        self.assertEqual(lm.call_count, 1)
        self.assertEqual(len(calls), 2 + 3)                             # 2 configured + 3 discovered, once each

    def test_listing_is_throttled_across_back_to_back_failures(self):
        with mock.patch.object(llm, "list_models", return_value=[]) as lm, \
                mock.patch.object(llm.requests, "post", side_effect=self.server(set(), [])):
            for _ in range(2):
                with self.assertRaises(llm.LLMError):
                    llm.generate_json("p")
        self.assertEqual(lm.call_count, 1)

    def test_not_triggered_by_other_failures(self):
        with mock.patch.object(llm, "list_models") as lm, \
                mock.patch.object(llm.requests, "post", return_value=err(503, "overloaded")):
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=2)
        lm.assert_not_called()

    def test_not_triggered_when_only_some_models_are_missing(self):
        def post(url, headers=None, json=None, timeout=None):
            return err(404, "gone") if json["model"] == "m-main" else err(503, "overloaded")
        with mock.patch.object(llm, "list_models") as lm, mock.patch.object(llm.requests, "post", side_effect=post):
            with self.assertRaises(llm.LLMError):
                llm.generate_json("p", max_attempts=2)
        lm.assert_not_called()


class TestRankModels(unittest.TestCase):
    def test_excludes_non_chat_models_and_already_tried(self):
        names = ["whisper-large-v3", "playai-tts", "meta-llama/llama-guard-4", "text-embedding-x",
                 "groq/compound", "llama-3.3-70b-test", "tried-model", "openai/gpt-oss-20b"]
        ranked = llm.rank_models(names, exclude={"tried-model"})
        self.assertEqual(ranked, ["llama-3.3-70b-test", "openai/gpt-oss-20b"])

    def test_preference_order_then_alphabetical(self):
        ranked = llm.rank_models(["zzz-model", "aaa-model", "qwen-x", "openai/gpt-oss-120b", "llama-3.1-8b-x"])
        self.assertEqual(ranked, ["openai/gpt-oss-120b", "qwen-x", "llama-3.1-8b-x", "aaa-model", "zzz-model"])


if __name__ == "__main__":
    unittest.main()
