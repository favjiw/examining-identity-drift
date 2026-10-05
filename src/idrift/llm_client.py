"""OpenAI-compatible chat client. Key from env only, never logged."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass
class ChatResult:
    """Single chat completion result + usage."""

    text: str
    usage_in: int = 0
    usage_out: int = 0
    latency_ms: int = 0
    finish_reason: str = "stop"
    model: str = ""


def _load_env_fallback(var_name: str) -> str:
    """Ambil dari os.environ; jika nihil coba baca .env."""
    val = os.environ.get(var_name, "")
    if val:
        return val
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k == var_name and v:
                    os.environ[k] = v
                    return v
        except Exception:
            pass
    return ""


@dataclass
class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    max_retries: int = 5
    timeout: float = 60.0
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> OpenAI:
        """Build client lazily so import/tests need no key."""
        if self._client is not None:
            return self._client
        api_key = _load_env_fallback(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env}")
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> ChatResult:
        """Send chat messages, retry 5xx/connection errors."""
        client = self._get_client()

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> ChatResult:
            start = time.monotonic()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = int((time.monotonic() - start) * 1000)
            choice = resp.choices[0]
            usage = resp.usage
            return ChatResult(
                text=(choice.message.content or ""),
                usage_in=int(getattr(usage, "prompt_tokens", 0) or 0),
                usage_out=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=latency,
                finish_reason=str(choice.finish_reason or "stop"),
                model=self.model,
            )

        return _call()

    def safe_params(self) -> dict[str, str]:
        """Log-safe params. No key value here."""
        return {"model": self.model, "api_key_env": self.api_key_env, "base_url": self.base_url or ""}
