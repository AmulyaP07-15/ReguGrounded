# utils/rate_limiter.py
#
# Token-bucket rate limiter for OpenAI API calls.
#
# Two limits are enforced:
#   RPM  — requests per minute  (OpenAI free-tier default: 3)
#   TPM  — tokens per minute    (OpenAI free-tier default: 40_000)
#
# Both limits are configurable via constructor args or environment variables:
#   REGUGROUNDED_RPM   (default: 60  — reasonable paid-tier value)
#   REGUGROUNDED_TPM   (default: 90_000)
#
# Usage (module-level singleton pattern):
#   from utils.rate_limiter import rate_limiter
#   rate_limiter.acquire(estimated_tokens=500)   # blocks if needed
#
# The limiter is also available as a decorator:
#   @rate_limited(estimated_tokens=500)
#   def call_openai(...): ...

import os
import threading
import time
from functools import wraps
from typing import Callable, Optional

from utils.logger import setup_logger

logger = setup_logger("rate_limiter")

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

_DEFAULT_RPM = int(os.getenv("REGUGROUNDED_RPM", "60"))
_DEFAULT_TPM = int(os.getenv("REGUGROUNDED_TPM", "90000"))

# How long to sleep between retry checks when the bucket is empty (seconds)
_POLL_INTERVAL = 0.05

# Maximum time to wait before raising RateLimitError (seconds)
_MAX_WAIT = 120.0


class RateLimitError(RuntimeError):
    """Raised when acquire() cannot obtain a token within MAX_WAIT seconds."""


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------

class _TokenBucket:
    """
    Thread-safe token bucket.

    Tokens refill continuously at `rate` tokens/second up to `capacity`.
    `acquire(tokens)` blocks until `tokens` are available, then deducts them.
    """

    def __init__(self, capacity: float, rate: float):
        self._capacity  = float(capacity)
        self._rate      = float(rate)         # tokens per second
        self._tokens    = float(capacity)     # start full
        self._last_tick = time.monotonic()
        self._lock      = threading.Lock()

    def _refill(self) -> None:
        now    = time.monotonic()
        delta  = now - self._last_tick
        self._tokens = min(self._capacity, self._tokens + delta * self._rate)
        self._last_tick = now

    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def acquire(self, tokens: float = 1.0, timeout: float = _MAX_WAIT) -> float:
        """
        Block until `tokens` are available and deduct them.
        Returns the time waited in seconds.
        Raises RateLimitError if timeout is exceeded.
        """
        deadline = time.monotonic() + timeout
        waited   = 0.0

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

            sleep_for = _POLL_INTERVAL
            if time.monotonic() + sleep_for > deadline:
                raise RateLimitError(
                    f"Rate limit: could not acquire {tokens:.0f} tokens "
                    f"within {timeout:.0f}s"
                )
            time.sleep(sleep_for)
            waited += sleep_for


# ---------------------------------------------------------------------------
# Composite rate limiter (RPM + TPM)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Enforces both a requests-per-minute and a tokens-per-minute limit.

    Internally uses two token buckets:
      - RPM bucket: 1 token per request, refills at rpm/60 tokens/sec
      - TPM bucket: tokens equal to token estimate, refills at tpm/60 tokens/sec
    """

    def __init__(
        self,
        rpm: int = _DEFAULT_RPM,
        tpm: int = _DEFAULT_TPM,
    ):
        self.rpm = rpm
        self.tpm = tpm
        # Buckets start full; rate = limit / 60 tokens per second
        self._rpm_bucket = _TokenBucket(capacity=rpm,   rate=rpm / 60.0)
        self._tpm_bucket = _TokenBucket(capacity=tpm,   rate=tpm / 60.0)
        self._lock        = threading.Lock()

        logger.info(
            f"RateLimiter initialised — RPM: {rpm}, TPM: {tpm:,}"
        )

    def acquire(self, estimated_tokens: int = 500) -> float:
        """
        Block until both RPM and TPM capacity is available.

        Args:
            estimated_tokens: Rough upper-bound of tokens this call will use.
                              Defaults to 500 (a typical small completion).

        Returns:
            Seconds waited (0.0 if no throttling was needed).
        """
        t0 = time.monotonic()

        # Acquire RPM token first (cheaper check)
        waited_rpm = self._rpm_bucket.acquire(tokens=1)
        if waited_rpm > 0.1:
            logger.warning(
                f"RPM throttle: waited {waited_rpm:.2f}s "
                f"(limit={self.rpm} req/min)"
            )

        # Then acquire TPM tokens
        waited_tpm = self._tpm_bucket.acquire(tokens=estimated_tokens)
        if waited_tpm > 0.1:
            logger.warning(
                f"TPM throttle: waited {waited_tpm:.2f}s "
                f"for {estimated_tokens} tokens (limit={self.tpm:,} tok/min)"
            )

        total_waited = time.monotonic() - t0
        if total_waited > 0.1:
            logger.debug(f"Rate limiter total wait: {total_waited:.2f}s")

        return total_waited

    def acquire_with_retry(
        self,
        call: Callable,
        estimated_tokens: int = 500,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Acquire rate-limit capacity, call `call()`, and retry on 429 errors.

        On a 429 the limiter resets the RPM bucket to near-empty (back-pressure)
        so subsequent calls slow down automatically.

        Args:
            call:             Zero-argument callable that makes the API request.
            estimated_tokens: Token estimate for rate-limiter pre-deduction.
            max_retries:      How many 429s to absorb before re-raising.
            retry_delay:      Base sleep between retries (doubles each time).

        Returns:
            Whatever `call()` returns.
        """
        for attempt in range(max_retries + 1):
            self.acquire(estimated_tokens)
            try:
                return call()
            except Exception as exc:
                # Detect OpenAI 429 / RateLimitError by name or status code
                is_429 = (
                    "429" in str(exc)
                    or "rate_limit" in str(exc).lower()
                    or "RateLimitError" in type(exc).__name__
                )
                if is_429 and attempt < max_retries:
                    sleep = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"OpenAI 429 on attempt {attempt + 1}/{max_retries + 1} — "
                        f"sleeping {sleep:.1f}s then retrying"
                    )
                    # Back-pressure: drain the RPM bucket so next acquire waits
                    self._rpm_bucket._tokens = 0
                    time.sleep(sleep)
                else:
                    raise

    def status(self) -> dict:
        """Return current bucket levels as a dict (useful for health checks)."""
        return {
            "rpm_available":   round(self._rpm_bucket.available(), 2),
            "rpm_limit":       self.rpm,
            "tpm_available":   round(self._tpm_bucket.available(), 2),
            "tpm_limit":       self.tpm,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def rate_limited(estimated_tokens: int = 500) -> Callable:
    """
    Decorator — acquires rate-limit capacity before each call.

    Usage:
        @rate_limited(estimated_tokens=800)
        def call_openai(self, prompt): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter.acquire(estimated_tokens)
            return func(*args, **kwargs)
        return wrapper
    return decorator
