"""
utils/llm_client.py — Unified LLM client (Anthropic / Groq / Gemini)
=========================================================================
Priority order:
  1. ANTHROPIC_API_KEY is set  →  use AnthropicAdapter (Claude)
  2. GROQ_API_KEY is set       →  use Groq (Llama 3.3 70B, free tier)
  3. GEMINI_API_KEY* are set   →  use RotatingGeminiClient (cycles on 429)

All adapters expose the same OpenAI-compatible interface:

    client.chat.completions.create(model=..., messages=..., temperature=..., ...)
    response.choices[0].message.content

so all existing call sites work unchanged.

Environment variables (set in .env):
    ANTHROPIC_API_KEY   — Claude API key (highest priority)
    ANTHROPIC_MODEL     — Claude model (default: claude-haiku-4-5-20251001)
    GROQ_API_KEY        — Groq API key (free tier, 14,400 req/day)
    GROQ_API_KEY_2      — second Groq key (optional, rotated on 429)
    GROQ_API_KEY_3      — third Groq key  (optional, rotated on 429)
    GROQ_MODEL          — Groq model (default: llama-3.3-70b-versatile)
    GEMINI_API_KEY      — first Gemini key
    GEMINI_API_KEY_2    — second Gemini key (optional)
    GEMINI_API_KEY_3    — third Gemini key  (optional)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

logger = logging.getLogger("llm_client")


# ── Shared fake response objects (mimic openai.types.chat.ChatCompletion) ──

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


# ── Anthropic adapter ────────────────────────────────────────────────────────

class _AnthropicCompletionsProxy:
    def __init__(self, adapter: "AnthropicAdapter"):
        self._a = adapter

    def create(self, **kwargs):
        return self._a._create(**kwargs)


class _AnthropicChatProxy:
    def __init__(self, adapter: "AnthropicAdapter"):
        self.completions = _AnthropicCompletionsProxy(adapter)


class AnthropicAdapter:
    """
    Wraps the Anthropic SDK to present the same interface as an OpenAI client.
    Handles response_format JSON mode via system prompt injection.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_CLAUDE_MODEL):
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model  = model
        self.chat    = _AnthropicChatProxy(self)
        logger.info(f"AnthropicAdapter initialised — model: {model}")

    def _create(self, **kwargs) -> _FakeResponse:
        # Strip / translate OpenAI-only params
        kwargs.pop("model", None)                        # we use self._model
        response_format = kwargs.pop("response_format", None)
        max_tokens      = kwargs.pop("max_tokens", 2048)
        temperature     = kwargs.pop("temperature", 0.1)
        messages        = kwargs.get("messages", [])

        # JSON mode: Anthropic doesn't have response_format, so inject a
        # system instruction instead.
        system = None
        if response_format and response_format.get("type") == "json_object":
            system = (
                "You must respond with valid JSON only. "
                "Do not include any text, explanation, or markdown outside the JSON object."
            )

        create_kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
        )
        if system:
            create_kwargs["system"] = system

        response = self._client.messages.create(**create_kwargs)
        content  = response.content[0].text
        return _FakeResponse(content)


# ── Rotating Gemini client ───────────────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "quota" in msg or "resource_exhausted" in msg


class _GeminiCompletionsProxy:
    def __init__(self, rotator: "RotatingGeminiClient"):
        self._r = rotator

    def create(self, **kwargs):
        return self._r._create(**kwargs)


class _GeminiChatProxy:
    def __init__(self, rotator: "RotatingGeminiClient"):
        self.completions = _GeminiCompletionsProxy(rotator)


class RotatingGeminiClient:
    """
    Drop-in replacement for an OpenAI client pointed at the Gemini endpoint.
    Exposes client.chat.completions.create(...) and rotates keys on 429.
    """

    def __init__(self):
        from openai import OpenAI

        env_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
        ]
        self._keys = [k.strip() for k in env_keys if k and k.strip()]
        if not self._keys:
            raise EnvironmentError(
                "No Gemini API key found. Set GEMINI_API_KEY in .env"
            )
        self._clients = [
            OpenAI(api_key=k, base_url=_GEMINI_BASE_URL) for k in self._keys
        ]
        self._idx  = 0
        self.chat  = _GeminiChatProxy(self)
        logger.info(f"RotatingGeminiClient initialised with {len(self._keys)} key(s)")

    def _create(self, **kwargs):
        start_idx = self._idx
        tried = 0
        while tried < len(self._clients):
            try:
                return self._clients[self._idx].chat.completions.create(**kwargs)
            except Exception as exc:
                if _is_quota_error(exc) and len(self._clients) > 1:
                    next_idx = (self._idx + 1) % len(self._clients)
                    if next_idx == start_idx:
                        raise
                    logger.warning(
                        f"Key {self._idx + 1}/{len(self._clients)} quota exhausted — "
                        f"rotating to key {next_idx + 1}"
                    )
                    self._idx = next_idx
                    tried += 1
                else:
                    raise
        raise Exception(
            f"All {len(self._clients)} Gemini API key(s) exhausted quota."
        )


# ── Groq client ──────────────────────────────────────────────────────────────

_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_BASE_URL      = "https://api.groq.com/openai/v1"
# Groq free tier: 30 RPM. We throttle to 25 RPM (2.4s between calls) to stay safe.
_GROQ_MIN_INTERVAL  = 2.4


class GroqClient:
    """
    Thin wrapper around the OpenAI client pointed at Groq's endpoint.
    Groq supports the full OpenAI chat completions interface including
    response_format JSON mode for llama-3.3-70b-versatile.
    Enforces a minimum interval between calls to stay within 30 RPM.
    """

    class _GroqCompletionsProxy:
        def __init__(self, groq: "GroqClient"):
            self._g = groq

        def create(self, **kwargs):
            import time
            kwargs = dict(kwargs)
            kwargs["model"] = self._g._model
            start_idx = self._g._idx
            tried = 0
            while tried < len(self._g._clients):
                # Throttle per-key to stay within 30 RPM
                elapsed = time.time() - self._g._last_calls[self._g._idx]
                if elapsed < _GROQ_MIN_INTERVAL:
                    time.sleep(_GROQ_MIN_INTERVAL - elapsed)
                self._g._last_calls[self._g._idx] = time.time()
                try:
                    return self._g._clients[self._g._idx].chat.completions.create(**kwargs)
                except Exception as exc:
                    if ("429" in str(exc) or "rate_limit" in str(exc).lower()) and len(self._g._clients) > 1:
                        next_idx = (self._g._idx + 1) % len(self._g._clients)
                        if next_idx == start_idx:
                            raise
                        logger.warning(
                            f"Groq key {self._g._idx + 1}/{len(self._g._clients)} "
                            f"rate-limited — rotating to key {next_idx + 1}"
                        )
                        self._g._idx = next_idx
                        tried += 1
                    elif "429" in str(exc) or "rate_limit" in str(exc).lower():
                        # Only one key — wait and retry
                        logger.warning("Groq 429 — waiting 60s before retry")
                        time.sleep(60)
                        self._g._last_calls[self._g._idx] = time.time()
                        return self._g._clients[self._g._idx].chat.completions.create(**kwargs)
                    else:
                        raise
            raise Exception("All Groq API keys rate-limited.")

    class _GroqChatProxy:
        def __init__(self, groq: "GroqClient"):
            self.completions = GroqClient._GroqCompletionsProxy(groq)

    def __init__(self, api_key: str, model: str = _DEFAULT_GROQ_MODEL):
        import time
        from openai import OpenAI
        env_keys = [
            api_key,
            os.getenv("GROQ_API_KEY_2", "").strip(),
            os.getenv("GROQ_API_KEY_3", "").strip(),
        ]
        keys = [k for k in env_keys if k]
        self._clients    = [OpenAI(api_key=k, base_url=_GROQ_BASE_URL) for k in keys]
        self._model      = model
        self._idx        = 0
        self._last_calls = [0.0] * len(self._clients)
        self.chat        = GroqClient._GroqChatProxy(self)
        logger.info(f"GroqClient initialised — {len(keys)} key(s), model: {model}, throttle: {_GROQ_MIN_INTERVAL}s/call")


# ── Public factory ───────────────────────────────────────────────────────────

_singleton = None


def get_llm_client():
    """
    Return a module-level singleton LLM client.
    Priority: Anthropic → Groq → Gemini (rotating).
    """
    global _singleton
    if _singleton is None:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        groq_key      = os.getenv("GROQ_API_KEY", "").strip()

        if anthropic_key:
            model     = os.getenv("ANTHROPIC_MODEL", _DEFAULT_CLAUDE_MODEL).strip()
            _singleton = AnthropicAdapter(api_key=anthropic_key, model=model)
        elif groq_key:
            model     = os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL).strip()
            _singleton = GroqClient(api_key=groq_key, model=model)
        else:
            _singleton = RotatingGeminiClient()
    return _singleton
