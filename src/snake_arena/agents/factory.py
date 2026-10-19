"""Factory for RL agents, LLM clients, and DecisionManagers.

Single place where construction policy lives: checkpoint resolution, LLM
enable/disable logic, and worker lifecycle. The UI and benchmark both go
through this module so configurations are interpreted identically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import AppConfig
from ..llm.ollama_client import OllamaClient
from ..llm.strategist import StrategyWorker
from ..utils import paths
from .decision_manager import DecisionManager
from .rl_agent import RLAgent

log = logging.getLogger("snake_arena.factory")


def resolve_checkpoint_path(config: AppConfig, name: str | None = None) -> Path:
    """Checkpoints may be absolute, project-relative ('models/x.pt'), or a
    bare filename (resolved into the models directory)."""
    raw = name or config.agent.rl_checkpoint
    path = Path(raw)
    if not path.is_absolute():
        if path.parent == Path("."):
            path = paths.models_dir() / path
        else:
            path = paths.project_root() / path
    return path


def build_rl_agent(config: AppConfig, load_checkpoint: bool = True) -> RLAgent:
    """RL agent; untrained (random init) when no checkpoint exists/loads."""
    agent = RLAgent(rl_cfg=config.rl)
    if load_checkpoint:
        path = resolve_checkpoint_path(config)
        if path.exists():
            try:
                payload = agent.load_checkpoint(path)
                log.info("Loaded RL checkpoint %s (%s)", path, describe_payload(payload))
            except Exception as exc:  # noqa: BLE001 - bad checkpoint must not crash
                log.error("Failed to load checkpoint %s: %s (using untrained agent)", path, exc)
        else:
            log.info("No RL checkpoint at %s; running with an untrained network.", path)
    return agent


def describe_payload(payload: dict) -> str:
    from ..rl.checkpoints import describe_checkpoint

    return describe_checkpoint(payload)


def build_llm_client(config: AppConfig) -> OllamaClient | None:
    if not config.llm.enabled:
        return None
    return OllamaClient(
        host=config.llm.host,
        model=config.llm.model,
        timeout=config.llm.timeout_seconds,
        connect_timeout=config.llm.connection_timeout_seconds,
        max_retries=config.llm.max_retries,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def build_strategy_worker(client: OllamaClient | None, config: AppConfig) -> StrategyWorker | None:
    if client is None:
        return None
    return StrategyWorker(client, min_interval=config.llm.min_interval_seconds)


def build_manager(
    mode: str,
    config: AppConfig,
    rl_agent: RLAgent | None = None,
    strategist_worker: StrategyWorker | None = None,
    llm_client: OllamaClient | None = None,
    validator=None,
) -> DecisionManager:
    """Build a DecisionManager for ``mode``.

    LLM wiring:
      * pass ``strategist_worker`` for the interactive app (background thread);
      * pass ``llm_client`` for headless benchmark runs (synchronous, throttled);
      * pass neither for a guaranteed offline run (deterministic fallback).
    """
    if mode in ("rl", "llm_rl", "hybrid"):
        if rl_agent is None:
            rl_agent = build_rl_agent(config)
    return DecisionManager(
        mode=mode,
        config=config,
        rl_agent=rl_agent,
        validator=validator,
        strategist_worker=strategist_worker,
        llm_client=llm_client,
    )
