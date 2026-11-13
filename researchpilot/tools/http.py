"""Shared GET helper: bounded retry/backoff and errors that never contain the URL.

Error messages deliberately omit the request URL because some services take the
API key as a query parameter (e.g. OpenAlex); keys must never reach logs or the UI.
"""
import time
from typing import Optional

import requests

from .. import config
from ..logging_setup import get_logger

log = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_QUOTA_STATUS = {402, 409, 429}
_sleep = time.sleep  # patched in tests


class ServiceError(Exception):
    """An external service failed (network, HTTP error, bad payload)."""

    def __init__(self, service: str, message: str, status: Optional[int] = None):
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status = status


class QuotaError(ServiceError):
    """Free quota / rate limit exhausted. Callers should degrade gracefully."""


def get_with_retry(service: str, url: str, params: Optional[dict] = None,
                   headers: Optional[dict] = None, max_attempts: Optional[int] = None) -> requests.Response:
    attempts = max_attempts or config.HTTP_MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers,
                                    timeout=config.HTTP_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == attempts:
                raise ServiceError(service, f"network error ({type(exc).__name__})") from None
            log.warning("%s network error attempt=%d/%d", service, attempt, attempts)
            _sleep(min(10, 2 ** attempt))
            continue
        except requests.RequestException as exc:
            raise ServiceError(service, f"request failed ({type(exc).__name__})") from None

        status = response.status_code
        if status in _RETRY_STATUS and attempt < attempts:
            retry_after = str(response.headers.get("Retry-After", ""))
            if status == 429 and retry_after.isdigit() and int(retry_after) > config.HTTP_MAX_RETRY_WAIT:
                # Waiting out a long rate-limit window would freeze the app; let the caller use other sources.
                raise QuotaError(service, f"rate limited; asked to wait {retry_after}s (HTTP 429)", 429)
            delay = min(30, int(retry_after)) if retry_after.isdigit() else min(10, 2 ** attempt)
            log.warning("%s HTTP %s attempt=%d/%d; retrying in %ss", service, status, attempt, attempts, delay)
            _sleep(delay)
            continue
        if status in _QUOTA_STATUS:
            raise QuotaError(service, f"quota or rate limit reached (HTTP {status})", status)
        if status >= 400:
            raise ServiceError(service, f"HTTP {status}", status)
        return response
    raise ServiceError(service, "request failed")  # defensive
