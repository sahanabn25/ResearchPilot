"""LLM access: provider-agnostic JSON calls with bounded retry, model fallback and health memory.

Providers (LLM_PROVIDER): groq (default, free tier), gemini, openrouter, ollama, openai.
All except gemini speak the OpenAI-compatible /chat/completions protocol and need no extra SDK
(only `requests`). The google-genai SDK is imported lazily and only for provider "gemini".

Never logs API keys or prompt text; error text is redacted (see safe_message).
"""
import json
import random
import re
import time
from typing import Optional

import requests

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)

_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_client = None
_sleep = time.sleep  # patched in tests

_KEY_PATTERNS = (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), re.compile(r"gsk_[0-9A-Za-z]{20,}"),
                 re.compile(r"sk-[0-9A-Za-z_\-]{20,}"))
_RETRY_HINT_RES = (re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE),
                   re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
                   re.compile(r"retry-after:?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE))
_TRY_AGAIN_RE = re.compile(r"try again in\s+([0-9hms.]+)", re.IGNORECASE)   # Groq: "try again in 1m3.6s" / "250ms"
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DAILY_QUOTA_RE = re.compile(r"per.?day|\b[tr]pd\b", re.IGNORECASE)


def safe_message(exc, limit: int = 300) -> str:
    """Error text that is safe to log or show: whitespace-collapsed, truncated, API keys redacted."""
    text = " ".join(str(exc).split())
    for key in (config.GEMINI_API_KEY, config.LLM_API_KEY):
        if key and len(key) >= 6:
            text = text.replace(key, "***")
    for pattern in _KEY_PATTERNS:
        text = pattern.sub("***", text)
    return text[:limit]


def _duration_seconds(token: str):
    parts = _DURATION_PART_RE.findall(token)
    if not parts:
        return None
    return sum(float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit] for num, unit in parts)


def retry_hint_seconds(exc):
    """Seconds the provider asked us to wait, capped at 60; None if no hint is present."""
    text = str(exc)
    for pattern in _RETRY_HINT_RES:
        match = pattern.search(text)
        if match:
            return min(60.0, float(match.group(1)))
    match = _TRY_AGAIN_RE.search(text)
    if match:
        seconds = _duration_seconds(match.group(1))
        if seconds is not None:
            return min(60.0, seconds)
    return None


class LLMError(Exception):
    """The LLM call failed after bounded retries (or failed non-transiently)."""


class LLMConfigError(LLMError, ValueError):
    """Missing / invalid LLM configuration (e.g. no API key)."""


class ProviderHTTPError(Exception):
    """HTTP error from an OpenAI-compatible provider. `.code` is the status (used by retry logic)."""

    def __init__(self, status: int, message: str, retry_after=None):
        extra = f" Retry-After: {retry_after}s." if retry_after else ""
        super().__init__(f"HTTP {status}: {message}{extra}")
        self.code = status


# ------------------------------------------------------------------ configuration helpers
def is_configured() -> bool:
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        return bool(config.GEMINI_API_KEY)
    if provider not in config.SUPPORTED_PROVIDERS:
        return False
    if provider == "ollama":
        return bool(config.LLM_BASE_URL and config.LLM_MODEL)
    return bool(config.LLM_API_KEY and config.LLM_BASE_URL and config.LLM_MODEL)


def missing_credentials_message() -> str:
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        return "GEMINI_API_KEY is missing from your .env file."
    if provider not in config.SUPPORTED_PROVIDERS:
        return f"Unknown LLM_PROVIDER {provider!r}. Use one of: {', '.join(config.SUPPORTED_PROVIDERS)}."
    preset = config._PRESETS[provider]
    if provider == "openai":
        return "For LLM_PROVIDER=openai set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in your .env file."
    if provider == "ollama":
        return "For LLM_PROVIDER=ollama make sure LLM_MODEL is set and Ollama is running."
    hint = " Create a free key at console.groq.com/keys." if provider == "groq" else ""
    return f"{preset['key_env']} is missing from your .env file.{hint}"


def describe() -> str:
    model = config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.LLM_MODEL
    return f"{config.LLM_PROVIDER} · {model or 'no model set'}"


def _model_chain() -> list:
    if config.LLM_PROVIDER == "gemini":
        primary, fallbacks = config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODELS
    else:
        primary, fallbacks = config.LLM_MODEL, config.LLM_FALLBACK_MODELS
    chain = [primary]
    chain += [m for m in fallbacks if m not in chain]
    return chain + [m for m in _auto_models if m not in chain]


# ------------------------------------------------------------------ failure classification and health memory
_NETWORK_ERROR_NAMES = {"TransportError", "RemoteProtocolError", "ReadTimeout", "ConnectTimeout", "ConnectError",
                        "ReadError", "WriteError", "PoolTimeout", "TimeoutException", "NetworkError",
                        "ProtocolError", "ChunkedEncodingError", "IncompleteRead", "ConnectionError", "Timeout"}


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT_CODES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # e.g. httpx.RemoteProtocolError "Server disconnected without sending a response"
    return any(cls.__name__ in _NETWORK_ERROR_NAMES for cls in type(exc).__mro__)


# Short-lived memory of models that just failed, so later calls start with a healthy model
# instead of re-paying the full retry wait every time (e.g. one wait per paper batch).
_unhealthy_until: dict = {}
_now = time.monotonic  # patched in tests


def reset_health() -> None:
    global _last_discovery
    _unhealthy_until.clear()
    _auto_models.clear()
    _last_discovery = None


def _mark_unhealthy(model: str, seconds: float) -> None:
    _unhealthy_until[model] = _now() + seconds


def _ordered_models() -> list:
    """Configured chain, healthy models first; cooling-down models stay as a last resort."""
    chain, t = _model_chain(), _now()
    return ([m for m in chain if _unhealthy_until.get(m, 0) <= t]
            + [m for m in chain if _unhealthy_until.get(m, 0) > t])


# ------------------------------------------------------------------ provider calls
def get_client():
    """Gemini SDK client (provider "gemini" only)."""
    global _client
    if not config.GEMINI_API_KEY:
        raise LLMConfigError("GEMINI_API_KEY is missing from your .env file.")
    if _client is None:
        from google import genai  # lazy import

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _call_gemini(prompt: str, system_instruction: str, temperature: float, model: str):
    from google.genai import types  # lazy import

    response = get_client().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction,
                                           response_mime_type="application/json", temperature=temperature))
    meta = getattr(response, "usage_metadata", None)
    usage = {"prompt_tokens": getattr(meta, "prompt_token_count", None),
             "output_tokens": getattr(meta, "candidates_token_count", None)}
    return response.text, usage


def _error_text(response) -> str:
    try:
        err = response.json().get("error")
        text = err.get("message") if isinstance(err, dict) else err
        if text:
            return " ".join(str(text).split())[:400]
    except Exception:
        pass
    return " ".join((getattr(response, "text", "") or "").split())[:300] or "no error details"


def _retry_after_header(response):
    value = str(getattr(response, "headers", {}).get("Retry-After", ""))
    return value if value.isdigit() else None


def _call_openai_compat(prompt: str, system_instruction: str, temperature: float, model: str):
    """Chat-completions call for Groq, OpenRouter, Ollama and other OpenAI-compatible servers."""
    if not config.LLM_BASE_URL:
        raise LLMConfigError("LLM_BASE_URL is not set.")
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "system", "content": system_instruction or "Return valid JSON only."},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}}

    response = requests.post(url, headers=headers, json=body, timeout=config.LLM_TIMEOUT_SECONDS)
    if response.status_code == 400 and re.search(r"response_format|json_object|json mode", _error_text(response),
                                                 re.IGNORECASE):
        # Some models/servers do not support JSON mode; our prompts already demand JSON, so retry without it.
        plain_body = {k: v for k, v in body.items() if k != "response_format"}
        response = requests.post(url, headers=headers, json=plain_body, timeout=config.LLM_TIMEOUT_SECONDS)
    if response.status_code >= 400:
        raise ProviderHTTPError(response.status_code, _error_text(response), _retry_after_header(response))
    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        raw_usage = data.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response format from the LLM provider") from exc
    return text, {"prompt_tokens": raw_usage.get("prompt_tokens"), "output_tokens": raw_usage.get("completion_tokens")}


def _call_model(prompt: str, system_instruction: str, temperature: float, model: Optional[str] = None):
    """Single LLM call. Returns (text, usage_dict). Patched in unit tests."""
    model = model or _model_chain()[0]
    if config.LLM_PROVIDER == "gemini":
        return _call_gemini(prompt, system_instruction, temperature, model)
    return _call_openai_compat(prompt, system_instruction, temperature, model)


def _extract_json(text: str):
    """Parse JSON, tolerating Markdown fences or prose around the object (common with open models)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        candidate = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise first


# ------------------------------------------------------------------ automatic model discovery
# Free-tier model catalogues change (models get renamed or retired). If EVERY configured model is
# "not found", ask the provider which models this key can use and pick suitable ones, instead of failing.
_EXCLUDE_WORDS = ("whisper", "tts", "guard", "embed", "orpheus", "playai", "moderation", "compound",
                  "safeguard", "rerank", "transcribe", "speech", "image", "audio")
_PREFERRED = ("llama-3.3-70b", "gpt-oss-120b", "llama-4", "qwen", "kimi", "gpt-oss-20b", "llama-3.1-8b",
              "llama", "gemma", "mistral")
_auto_models: list = []          # models found by discovery; appended to the configured chain
_last_discovery = None           # monotonic time of the last discovery attempt


def rank_models(names, exclude=()) -> list:
    """Usable text-generation models, best guess first (audio/guard/embedding models are excluded)."""
    usable = [n for n in names if n not in exclude and not any(w in n.lower() for w in _EXCLUDE_WORDS)]

    def score(name):
        low = name.lower()
        for i, preferred in enumerate(_PREFERRED):
            if preferred in low:
                return (i, name)
        return (len(_PREFERRED), name)
    return sorted(usable, key=score)


def _discover_alternatives(exclude) -> list:
    global _last_discovery
    if _last_discovery is not None and _now() - _last_discovery < 60:
        return []                      # tried very recently; do not hammer the models endpoint
    _last_discovery = _now()
    try:
        names = list_models()
    except Exception as exc:
        log.warning("could not list available models: %s: %s", type(exc).__name__, safe_message(exc))
        return []
    ranked = rank_models(names, exclude)[:3]
    for model in ranked:
        if model not in _auto_models:
            _auto_models.append(model)
    if ranked:
        log.warning("configured models not available; auto-selected %s from the provider's model list. "
                    "Set LLM_MODEL in .env to make this permanent.", ranked)
    return ranked


class _AllModelsNotFound(Exception):
    def __init__(self, models, last_error):
        super().__init__(f"none of {models} exist for this key")
        self.models, self.last_error = models, last_error


def _not_found_message(tried) -> str:
    return (f"None of the models {list(tried)} are available for your {config.LLM_PROVIDER} API key, and no usable "
            "model could be found automatically. Run  python scripts/check_llm.py  to see the models your key can "
            "use, then set LLM_MODEL in your .env file.")


# ------------------------------------------------------------------ main entry point
def _try_models(models, prompt: str, system_instruction: str, temperature: float, attempts: int):
    last_error = None
    not_found = set()
    for model in models:
        # Primary gets the full retry budget; fallbacks are capped so a long outage
        # cannot multiply latency and cost.
        model_attempts = attempts if model == models[0] else min(attempts, 2)
        for attempt in range(1, model_attempts + 1):
            started = time.perf_counter()
            try:
                text, usage = _call_model(prompt, system_instruction, temperature, model)
                if not text:
                    raise ValueError("The LLM returned an empty response.")
                data = _extract_json(text)
                log.info("llm ok provider=%s model=%s attempt=%d latency=%.2fs tokens=%s",
                         config.LLM_PROVIDER, model, attempt, time.perf_counter() - started, usage)
                _unhealthy_until.pop(model, None)
                return data
            except LLMConfigError:
                raise
            except ValueError as exc:  # empty response or malformed JSON
                last_error = exc
                log.warning("llm bad output model=%s attempt=%d/%d: %s: %s",
                            model, attempt, model_attempts, type(exc).__name__, safe_message(exc))
            except Exception as exc:  # provider / network errors
                last_error = exc
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code == 404:
                    log.warning("llm model not found: %s; trying next model", model)
                    _mark_unhealthy(model, 600)
                    not_found.add(model)
                    break
                if code == 429 and _DAILY_QUOTA_RE.search(str(exc)):
                    # A per-DAY quota will not recover within our retry window; other models have their own quota.
                    log.warning("llm daily quota exhausted for model=%s (%s); trying next model",
                                model, safe_message(exc))
                    _mark_unhealthy(model, 1800)
                    break
                if not _is_transient(exc):
                    log.error("llm non-transient error model=%s: %s: %s", model, type(exc).__name__, safe_message(exc))
                    raise LLMError(f"LLM request failed: {type(exc).__name__}: {safe_message(exc)}") from exc
                log.warning("llm transient error model=%s attempt=%d/%d: %s: %s",
                            model, attempt, model_attempts, type(exc).__name__, safe_message(exc))
                if code == 503 and attempt >= 2 and model != models[-1]:
                    # "high demand" is model-specific; another model may answer right away.
                    log.warning("llm model %s overloaded; trying next model", model)
                    _mark_unhealthy(model, 90)
                    break

            if attempt < model_attempts:
                hint = retry_hint_seconds(last_error)
                base = hint + 1.0 if hint is not None else min(60, 3 * 2 ** attempt)  # default ~6s, 12s, 24s ...
                _sleep(base + random.uniform(0, 1))
        else:  # retries exhausted without success or an early switch
            _mark_unhealthy(model, 60)
        if model != models[-1]:
            log.warning("llm switching from %s to next fallback model", model)

    if not_found >= set(models):
        raise _AllModelsNotFound(list(models), last_error)
    raise LLMError(
        f"LLM request failed after trying {list(models)} ({attempts} attempts each). Last error: "
        f"{type(last_error).__name__}: {safe_message(last_error)}"
    )


def generate_json(prompt: str, system_instruction: str = "", temperature: float = 0.2,
                  max_attempts: Optional[int] = None):
    """Call the configured LLM in JSON mode and return the parsed JSON value.

    For each model in [primary + fallbacks] (healthy ones first): retries (bounded, with backoff and the
    provider's own wait hint) on transient errors (429/5xx/network drops), empty replies and malformed JSON.
    404 (model not found), a per-day quota, or repeated overload moves on to the next model. If EVERY model
    is "not found", the provider's model list is fetched and suitable models are tried automatically.
    Other errors (400/401/403...) fail immediately. Never logs API keys or prompt text.
    """
    if not is_configured():
        raise LLMConfigError(missing_credentials_message())

    attempts = max_attempts or config.LLM_MAX_ATTEMPTS
    models = _ordered_models()
    try:
        return _try_models(models, prompt, system_instruction, temperature, attempts)
    except _AllModelsNotFound as exc:
        alternatives = _discover_alternatives(exclude=set(models))
        if not alternatives:
            raise LLMError(_not_found_message(models)) from exc.last_error
        try:
            return _try_models(alternatives, prompt, system_instruction, temperature, attempts)
        except _AllModelsNotFound as exc2:
            raise LLMError(_not_found_message(models + alternatives)) from exc2.last_error


def list_models() -> list:
    """Model names the current provider/key can use."""
    if config.LLM_PROVIDER == "gemini":
        names = []
        for m in get_client().models.list():
            actions = getattr(m, "supported_actions", None)
            if actions is None or "generateContent" in actions:
                names.append(m.name.replace("models/", ""))
        return sorted(names)
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
    response = requests.get(config.LLM_BASE_URL.rstrip("/") + "/models", headers=headers,
                            timeout=config.LLM_TIMEOUT_SECONDS)
    if response.status_code >= 400:
        raise ProviderHTTPError(response.status_code, _error_text(response))
    return sorted(m["id"] for m in response.json().get("data", []) if isinstance(m, dict) and "id" in m)


if __name__ == "__main__":  # python -m researchpilot.llm
    print("Provider:", describe())
    print("Fallbacks:", _model_chain()[1:] or "(none)")
    print("\nModels available to your key:")
    for name in list_models():
        print("  ", name)
