"""Ollama HTTP client (stdlib ``urllib`` only -- no extra dependencies).

Provides connectivity checks, model listing, and JSON-mode chat calls with
timeouts and bounded retries. A pluggable ``transport`` callable allows tests
to inject mocked responses without any network access. All failures surface
as typed exceptions so callers can degrade gracefully.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Callable

log = logging.getLogger("snake_arena.llm.client")

# transport(method_path, payload_bytes, timeout) -> (status:int, body:bytes)
Transport = Callable[[str, bytes | None, float], tuple[int, bytes]]


class OllamaError(Exception):
    """Base class for all Ollama integration failures."""


class OllamaUnavailable(OllamaError):
    pass


class OllamaTimeout(OllamaError):
    pass


class OllamaBadResponse(OllamaError):
    pass


def _default_transport(base_url: str) -> Transport:
    def transport(path: str, payload: bytes | None, timeout: float) -> tuple[int, bytes]:
        url = base_url + path
        data = payload if payload is not None else (b"" if path.startswith("/api/tags") else None)
        req = urllib.request.Request(
            url,
            data=data,
            method="POST" if payload is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {base_url}: {exc}") from exc

    return transport


class OllamaClient:
    """Small, synchronous client; callers own threading (see StrategyWorker)."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b",
        timeout: float = 8.0,
        connect_timeout: float = 2.5,
        max_retries: int = 1,
        temperature: float = 0.4,
        max_tokens: int = 220,
        transport: Transport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._transport = transport or _default_transport(self.host)
        self.last_latency_ms: float = 0.0
        self.last_error: str | None = None

    # -- connectivity ---------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        """(available, detail): cheap GET /api/tags probe."""
        try:
            status, body = self._transport("/api/tags", None, self.connect_timeout)
        except OllamaError as exc:
            self.last_error = str(exc)
            return False, self.last_error
        if status != 200:
            detail = f"HTTP {status} from /api/tags"
            self.last_error = detail
            return False, detail
        try:
            data = json.loads(body.decode("utf-8"))
            names = [m.get("name", "?") for m in data.get("models", [])]
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = "Ollama responded but the model list was unreadable"
            self.last_error = detail
            return False, detail
        return True, f"ok; models: {', '.join(names) if names else '(none pulled yet)'}"

    def list_models(self) -> list[str]:
        status, body = self._transport("/api/tags", None, self.connect_timeout)
        if status != 200:
            raise OllamaUnavailable(f"HTTP {status} from /api/tags")
        data = json.loads(body.decode("utf-8"))
        return [m.get("name", "?") for m in data.get("models", [])]

    def has_model(self, model: str | None = None) -> tuple[bool, str]:
        model = model or self.model
        try:
            names = self.list_models()
        except OllamaError as exc:
            return False, str(exc)
        base = model.split(":")[0]
        for name in names:
            if name == model or name.split(":")[0] == base:
                return True, name
        return False, f"model '{model}' not pulled (run: ollama pull {model})"

    # -- inference --------------------------------------------------------------

    def chat_raw(self, system: str, user: str) -> str:
        """One chat completion; returns assistant text. Retries within
        ``max_retries`` on transport failures and malformed responses, then
        raises the last error."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        attempts = self.max_retries + 1
        last_exc: OllamaError | None = None
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                status, raw = self._transport("/api/chat", body, self.timeout)
            except OllamaError as exc:
                last_exc = exc
                self.last_error = str(exc)
                continue
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            if status != 200:
                last_exc = OllamaBadResponse(f"HTTP {status} from /api/chat: {raw[:200]!r}")
                self.last_error = str(last_exc)
                continue
            try:
                data = json.loads(raw.decode("utf-8"))
                content = data["message"]["content"]
                if not isinstance(content, str):
                    raise KeyError("content is not a string")
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
                last_exc = OllamaBadResponse(f"unreadable chat response: {exc}")
                self.last_error = str(last_exc)
                continue
            return content
        raise last_exc or OllamaError("chat failed")

    def chat_json(self, system: str, user: str) -> dict:
        from .schemas import extract_json_object

        text = self.chat_raw(system, user)
        return extract_json_object(text)
