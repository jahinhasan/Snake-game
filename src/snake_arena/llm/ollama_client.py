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
from collections.abc import Callable
from typing import Any

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
        fallback_models: list[str] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.requested_model = model
        self.model = model
        self.fallback_models = list(fallback_models) if fallback_models is not None \
            else ["qwen3:1.7b", "qwen3:0.6b"]
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._transport = transport or _default_transport(self.host)
        self.last_latency_ms: float = 0.0
        self.last_error: str | None = None
        self.model_note: str | None = None  # set when a fallback was picked
        self._model_resolved = False

    # -- model selection ---------------------------------------------------------

    @staticmethod
    def _match(names: list[str], wanted: str) -> str | None:
        """Exact tag match for ``wanted`` (e.g. ``qwen3:4b``), else None."""
        for name in names:
            if name == wanted:
                return name
        return None

    def resolve_model(self, force: bool = False) -> str:
        """Pick the best installed model: requested, then fallbacks, then any
        installed tag from the requested model's family (``qwen3:*``).

        Server-unreachable or list-unreadable cases keep the requested model;
        the actual chat call then fails exactly as it did before this existed.
        Runs once per client unless ``force`` (e.g. after a model-missing 404).
        """
        if self._model_resolved and not force:
            return self.model
        self._model_resolved = True
        try:
            names = self.list_models()
        except OllamaError as exc:
            log.debug("model resolution skipped: %s", exc)
            return self.model
        if self._match(names, self.model):
            return self.model
        for candidate in self.fallback_models:
            if candidate == self.requested_model:
                continue
            found = self._match(names, candidate)
            if found:
                self.model_note = (f"requested '{self.requested_model}' is not installed; "
                                   f"using smaller fallback '{found}'")
                log.info("%s", self.model_note)
                self.model = found
                return self.model
        family = self.requested_model.split(":")[0]
        for name in names:
            if name.split(":")[0] == family and name != self.requested_model:
                self.model_note = (f"requested '{self.requested_model}' is not installed; "
                                   f"using same-family model '{name}'")
                log.info("%s", self.model_note)
                self.model = name
                return self.model
        return self.model

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
        """(installed, detail): reports the *actual* installed tag, preferring
        an exact match, then a same-family tag (which callers should treat as
        "usable, but different from what was asked for")."""
        model = model or self.model
        try:
            names = self.list_models()
        except OllamaError as exc:
            return False, str(exc)
        exact = self._match(names, model)
        if exact:
            return True, exact
        family = model.split(":")[0]
        for name in names:
            if name.split(":")[0] == family:
                return True, f"{name} (family match; exact '{model}' not pulled)"
        return False, f"model '{model}' not pulled (run: ollama pull {model})"

    # -- inference --------------------------------------------------------------

    def chat_raw(self, system: str, user: str) -> str:
        """One chat completion; returns assistant text. Retries within
        ``max_retries`` on transport failures and malformed responses, then
        raises the last error. Before the first call (and after a model-level
        HTTP failure) the model is resolved against the installed tags, so a
        missing large model transparently falls back to a smaller one."""
        self.resolve_model()
        attempts = self.max_retries + 1
        last_exc: OllamaError | None = None
        for attempt in range(attempts):
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
                # 404 usually means "model not installed/removed": re-resolve
                # so the next attempt (if retries remain) can use a fallback.
                if status == 404:
                    self.resolve_model(force=True)
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
