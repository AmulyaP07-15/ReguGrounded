# tests/test_rate_limiter.py

import time

import pytest

from utils.rate_limiter import _TokenBucket, RateLimiter, RateLimitError, rate_limiter


# ---------------------------------------------------------------------------
# _TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_starts_full(self):
        b = _TokenBucket(capacity=10, rate=1)
        assert b.available() == pytest.approx(10, abs=0.1)

    def test_acquire_deducts_tokens(self):
        b = _TokenBucket(capacity=10, rate=0)   # no refill
        b.acquire(tokens=3)
        assert b.available() == pytest.approx(7, abs=0.1)

    def test_acquire_blocks_until_refill(self):
        # 1 token capacity, 20 tokens/sec refill — should wait ~0.05s for 1 token
        b = _TokenBucket(capacity=1, rate=20)
        b.acquire(tokens=1)    # drain
        t0 = time.monotonic()
        b.acquire(tokens=1)    # should wait ~50 ms
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.03   # waited at least a bit

    def test_timeout_raises(self):
        b = _TokenBucket(capacity=1, rate=0)   # never refills
        b.acquire(tokens=1)    # drain
        with pytest.raises(RateLimitError):
            b.acquire(tokens=1, timeout=0.1)

    def test_no_wait_when_tokens_available(self):
        b = _TokenBucket(capacity=100, rate=10)
        t0 = time.monotonic()
        b.acquire(tokens=5)
        assert time.monotonic() - t0 < 0.05   # near-instant


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_acquire_no_throttle(self):
        rl = RateLimiter(rpm=600, tpm=900_000)   # very generous limits
        waited = rl.acquire(estimated_tokens=100)
        assert waited < 0.05

    def test_status_has_expected_keys(self):
        rl = RateLimiter(rpm=60, tpm=90_000)
        s = rl.status()
        assert "rpm_available" in s
        assert "tpm_available" in s
        assert s["rpm_limit"]  == 60
        assert s["tpm_limit"]  == 90_000

    def test_tpm_throttle(self):
        # Tiny TPM bucket so large request must wait
        rl = RateLimiter(rpm=600, tpm=100)   # 100 tok/min = ~1.67 tok/sec
        # Drain the bucket
        rl._tpm_bucket._tokens = 0
        t0 = time.monotonic()
        # Requesting 5 tokens at 1.67 tok/sec → ~3s; cap timeout low for test
        with pytest.raises(RateLimitError):
            rl._tpm_bucket.acquire(tokens=50, timeout=0.2)

    def test_acquire_with_retry_success(self):
        rl = RateLimiter(rpm=600, tpm=900_000)
        calls = []

        def fake_call():
            calls.append(1)
            return "ok"

        result = rl.acquire_with_retry(fake_call, estimated_tokens=10)
        assert result == "ok"
        assert len(calls) == 1

    def test_acquire_with_retry_on_429(self):
        rl = RateLimiter(rpm=600, tpm=900_000)
        attempts = []

        class Fake429(Exception):
            pass

        def fake_call():
            attempts.append(1)
            if len(attempts) < 3:
                raise Fake429("429 rate_limit exceeded")
            return "success_after_retries"

        result = rl.acquire_with_retry(
            fake_call,
            estimated_tokens=10,
            max_retries=3,
            retry_delay=0.01,
        )
        assert result == "success_after_retries"
        assert len(attempts) == 3

    def test_acquire_with_retry_exhausted(self):
        rl = RateLimiter(rpm=600, tpm=900_000)

        def always_429():
            raise Exception("429 rate_limit exceeded")

        with pytest.raises(Exception, match="429"):
            rl.acquire_with_retry(
                always_429,
                estimated_tokens=10,
                max_retries=2,
                retry_delay=0.01,
            )

    def test_non_429_not_retried(self):
        rl = RateLimiter(rpm=600, tpm=900_000)
        calls = []

        def bad_call():
            calls.append(1)
            raise ValueError("something else entirely")

        with pytest.raises(ValueError):
            rl.acquire_with_retry(bad_call, estimated_tokens=10, max_retries=3, retry_delay=0.01)
        assert len(calls) == 1   # no retry


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_singleton_exists(self):
        assert rate_limiter is not None

    def test_singleton_is_rate_limiter(self):
        assert isinstance(rate_limiter, RateLimiter)

    def test_singleton_has_positive_limits(self):
        assert rate_limiter.rpm > 0
        assert rate_limiter.tpm > 0
