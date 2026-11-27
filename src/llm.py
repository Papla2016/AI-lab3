"""LLM abstraction and OpenAI-compatible implementation."""
import json
import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from time import monotonic, sleep
from typing import Protocol

from openai import APIConnectionError, APIStatusError

DEFAULT_REQUESTS_PER_MINUTE = 12
MAX_RETRIES = 2


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class LLMRateLimitExceeded(RuntimeError):
    """Safe to display after the provider rejects a request with HTTP 429."""

    def __init__(self, daily: bool) -> None:
        message = (
            "HTTP 429. Исчерпана суточная квота модели. Возобновите прогон после её обновления."
            if daily else
            "HTTP 429. Временное ограничение запросов сохранилось после повторов. Возобновите прогон позже."
        )
        super().__init__(message)


class ModelRateLimiter:
    """Space requests evenly and share cooldowns between callers in this process."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.last_request: float | None = None
        self.cooldown_until = 0.0

    def acquire(self, rpm: int) -> None:
        while True:
            with self.lock:
                now = monotonic()
                next_request = self.last_request + 60 / rpm if self.last_request is not None else now
                wait = max(next_request, self.cooldown_until) - now
                if wait <= 0:
                    self.last_request = now
                    return
            sleep(wait)

    def defer(self, delay: float) -> None:
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, monotonic() + delay)


_limiters: dict[str, ModelRateLimiter] = {}
_limiters_lock = Lock()


def _retry_after(exc: APIStatusError) -> float:
    headers = exc.response.headers
    value = headers.get("Retry-After")
    if value is not None:
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = math.nan
        if math.isfinite(delay):
            return max(0, delay)
    try:
        delay = float(headers.get("retry-after-ms", "0")) / 1000
    except ValueError:
        return 0
    return max(0, delay) if math.isfinite(delay) else 0


def _daily_quota(exc: APIStatusError) -> bool:
    # Gemini includes the quota ID/metric in structured error details when available.
    body = json.dumps(exc.body, default=str).lower()
    daily = r"per[ _/-]*(?:day|daily)|daily[ _-]*(?:quota|limit|requests)|requests[ _-]*per[ _-]*day|\brpd\b"
    return re.search(daily, body) is not None


class OpenAICompatibleLLM:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=90.0)
        self.model = model
        self.requests_per_minute = requests_per_minute
        with _limiters_lock:
            self.limiter = _limiters.setdefault(model.removeprefix("models/").lower(), ModelRateLimiter())

    def generate(self, system: str, user: str) -> str:
        for attempt in range(MAX_RETRIES + 1):
            self.limiter.acquire(self.requests_per_minute)
            try:
                response = self.client.chat.completions.create(
                    model=self.model, temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
            except APIStatusError as exc:
                delay = max(2 ** attempt, _retry_after(exc))
                if exc.status_code == 429:
                    if _daily_quota(exc):
                        raise LLMRateLimitExceeded(daily=True) from None
                    self.limiter.defer(delay)
                    if attempt == MAX_RETRIES:
                        raise LLMRateLimitExceeded(daily=False) from None
                elif attempt == MAX_RETRIES or (exc.status_code not in (408, 409) and exc.status_code < 500):
                    raise
                else:
                    self.limiter.defer(delay)
            except APIConnectionError:
                if attempt == MAX_RETRIES:
                    raise
                self.limiter.defer(2 ** attempt)
            else:
                return response.choices[0].message.content or "{}"
