"""Application configuration.

Configuration lives in ``config/default.json`` and is loaded into typed
dataclasses with clamping/validation so that a corrupt or hand-edited file can
never crash the app: invalid values fall back to defaults with a warning.

User-facing edits (Settings screen) mutate the dataclasses in memory and are
persisted with :meth:`AppConfig.save`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import paths

log = logging.getLogger("snake_arena.config")

_SCHEMA_VERSION = 1


def _clamp(value: float, lo: float, hi: float, name: str) -> float:
    v = max(lo, min(hi, float(value)))
    if v != value:
        log.warning("Config value %s=%s out of range [%s, %s]; clamped to %s", name, value, lo, hi, v)
    return v


def _clamp_int(value: int, lo: int, hi: int, name: str) -> int:
    v = int(max(lo, min(hi, int(value))))
    if v != value:
        log.warning("Config value %s=%s out of range [%s, %s]; clamped to %s", name, value, lo, hi, v)
    return v


@dataclass
class BoardConfig:
    width: int = 24
    height: int = 18
    min_width: int = 10
    min_height: int = 10
    max_width: int = 40
    max_height: int = 30

    def validate(self) -> None:
        self.width = _clamp_int(self.width, self.min_width, self.max_width, "board.width")
        self.height = _clamp_int(self.height, self.min_height, self.max_height, "board.height")


@dataclass
class GameConfig:
    moves_per_second: int = 10
    min_moves_per_second: int = 2
    max_moves_per_second: int = 30
    max_foods: int = 4
    time_limit_seconds: int = 300
    seed: int | None = None

    def validate(self) -> None:
        self.moves_per_second = _clamp_int(
            self.moves_per_second, self.min_moves_per_second, self.max_moves_per_second, "game.moves_per_second"
        )
        self.max_foods = _clamp_int(self.max_foods, 0, 12, "game.max_foods")
        self.time_limit_seconds = _clamp_int(self.time_limit_seconds, 30, 3600, "game.time_limit_seconds")


@dataclass
class WindowConfig:
    width: int = 1280
    height: int = 800
    min_width: int = 1024
    min_height: int = 680
    fullscreen: bool = False
    fps_counter: bool = True

    def validate(self) -> None:
        self.width = _clamp_int(self.width, self.min_width, 4096, "window.width")
        self.height = _clamp_int(self.height, self.min_height, 4096, "window.height")


@dataclass
class AudioConfig:
    enabled: bool = True
    volume: float = 0.5

    def validate(self) -> None:
        self.volume = _clamp(self.volume, 0.0, 1.0, "audio.volume")


@dataclass
class VisualConfig:
    show_grid: bool = True
    show_path: bool = True
    show_danger: bool = True
    show_target: bool = True
    show_rejected: bool = True
    show_qvalues: bool = True
    smooth_movement: bool = True


@dataclass
class AgentConfig:
    type: str = "astar"
    opponent_type: str = "rule_based"
    rl_checkpoint: str = "models/dqn_default.pt"

    VALID_TYPES = (
        "rule_based",
        "astar",
        "bfs",
        "rl",
        "llm_astar",
        "llm_rl",
        "hybrid",
    )

    def validate(self) -> None:
        if self.type not in self.VALID_TYPES:
            log.warning("Unknown agent type %r; falling back to 'astar'.", self.type)
            self.type = "astar"
        if self.opponent_type not in self.VALID_TYPES:
            log.warning("Unknown opponent type %r; falling back to 'rule_based'.", self.opponent_type)
            self.opponent_type = "rule_based"


@dataclass
class LLMConfig:
    enabled: bool = True
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    timeout_seconds: float = 8.0
    connection_timeout_seconds: float = 2.5
    min_interval_seconds: float = 3.0
    max_retries: int = 1
    temperature: float = 0.4
    max_tokens: int = 220
    analysis_max_tokens: int = 500

    def validate(self) -> None:
        self.timeout_seconds = _clamp(self.timeout_seconds, 1.0, 60.0, "llm.timeout_seconds")
        self.connection_timeout_seconds = _clamp(
            self.connection_timeout_seconds, 0.5, 30.0, "llm.connection_timeout_seconds"
        )
        self.min_interval_seconds = _clamp(self.min_interval_seconds, 0.5, 120.0, "llm.min_interval_seconds")
        self.max_retries = _clamp_int(self.max_retries, 0, 3, "llm.max_retries")
        self.temperature = _clamp(self.temperature, 0.0, 1.5, "llm.temperature")
        self.max_tokens = _clamp_int(self.max_tokens, 40, 2048, "llm.max_tokens")
        self.analysis_max_tokens = _clamp_int(self.analysis_max_tokens, 40, 4096, "llm.analysis_max_tokens")
        host = str(self.host).strip()
        if not host.startswith(("http://", "https://")):
            log.warning("LLM host %r is not an HTTP URL; using default.", self.host)
            host = "http://127.0.0.1:11434"
        self.host = host.rstrip("/")
        if not str(self.model).strip():
            self.model = "qwen3:4b"


@dataclass
class RLRewardConfig:
    food_scale: float = 0.1
    trap_penalty: float = -2.0
    death_wall: float = -10.0
    death_self: float = -10.0
    death_opponent: float = -10.0
    step_cost: float = 0.0
    progress_scale: float = 0.12
    stagnation_start_steps: int = 120
    stagnation_penalty: float = 0.02
    win_bonus: float = 5.0


@dataclass
class RLConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    gamma: float = 0.90
    learning_rate: float = 0.0005
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_steps: int = 1000
    train_every_steps: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 25000
    double_dqn: bool = True
    max_episode_steps: int = 1200
    reward: RLRewardConfig = field(default_factory=RLRewardConfig)

    def validate(self) -> None:
        self.gamma = _clamp(self.gamma, 0.1, 0.999, "rl.gamma")
        self.learning_rate = _clamp(self.learning_rate, 1e-5, 0.1, "rl.learning_rate")
        self.buffer_size = _clamp_int(self.buffer_size, 1000, 2_000_000, "rl.buffer_size")
        self.batch_size = _clamp_int(self.batch_size, 8, 1024, "rl.batch_size")
        self.target_update_steps = _clamp_int(self.target_update_steps, 50, 100000, "rl.target_update_steps")
        self.train_every_steps = _clamp_int(self.train_every_steps, 1, 16, "rl.train_every_steps")
        self.eps_decay_steps = _clamp_int(self.eps_decay_steps, 100, 2_000_000, "rl.eps_decay_steps")
        self.max_episode_steps = _clamp_int(self.max_episode_steps, 100, 100000, "rl.max_episode_steps")
        if len(self.hidden_sizes) not in (1, 2, 3) or any(h < 16 for h in self.hidden_sizes):
            log.warning("Invalid rl.hidden_sizes %s; using [128, 128].", self.hidden_sizes)
            self.hidden_sizes = [128, 128]


@dataclass
class SafetyConfig:
    dead_end_check: bool = True
    min_escape_area_factor: float = 1.0
    flood_fill_limit: int = 900

    def validate(self) -> None:
        self.min_escape_area_factor = _clamp(
            self.min_escape_area_factor, 0.0, 3.0, "safety.min_escape_area_factor"
        )
        self.flood_fill_limit = _clamp_int(self.flood_fill_limit, 50, 20000, "safety.flood_fill_limit")


@dataclass
class HybridDecisionConfig:
    w_rl: float = 0.55
    w_path: float = 0.30
    w_space: float = 0.15

    def validate(self) -> None:
        total = self.w_rl + self.w_path + self.w_space
        if total <= 0:
            log.warning("Hybrid weights sum to %s; using defaults.", total)
            self.w_rl, self.w_path, self.w_space = 0.55, 0.30, 0.15
            return
        # Normalize so the documented convex combination stays valid.
        self.w_rl = self.w_rl / total
        self.w_path = self.w_path / total
        self.w_space = self.w_space / total


@dataclass
class AStarDecisionConfig:
    turn_penalty: float = 0.02
    risk_aversion: float = 0.6
    value_weight: float = 1.0
    survival_escape_ratio: float = 1.1

    def validate(self) -> None:
        self.turn_penalty = _clamp(self.turn_penalty, 0.0, 0.5, "decision.astar.turn_penalty")
        self.risk_aversion = _clamp(self.risk_aversion, 0.0, 2.0, "decision.astar.risk_aversion")
        self.value_weight = _clamp(self.value_weight, 0.0, 5.0, "decision.astar.value_weight")
        self.survival_escape_ratio = _clamp(
            self.survival_escape_ratio, 1.0, 3.0, "decision.astar.survival_escape_ratio"
        )


@dataclass
class DecisionConfig:
    hybrid: HybridDecisionConfig = field(default_factory=HybridDecisionConfig)
    astar: AStarDecisionConfig = field(default_factory=AStarDecisionConfig)

    def validate(self) -> None:
        self.hybrid.validate()
        self.astar.validate()


@dataclass
class BenchmarkConfig:
    episodes_per_agent: int = 20
    llm_episodes_per_agent: int = 5
    board_preset: str = "default"
    base_seed: int = 1000
    max_ticks_per_episode: int = 1500
    agents: list[str] = field(
        default_factory=lambda: [
            "rule_based",
            "astar",
            "bfs",
            "rl",
            "llm_astar",
            "llm_rl",
            "hybrid",
        ]
    )

    def validate(self) -> None:
        self.episodes_per_agent = _clamp_int(self.episodes_per_agent, 1, 500, "benchmark.episodes_per_agent")
        self.llm_episodes_per_agent = _clamp_int(
            self.llm_episodes_per_agent, 0, 200, "benchmark.llm_episodes_per_agent"
        )
        self.max_ticks_per_episode = _clamp_int(
            self.max_ticks_per_episode, 100, 20000, "benchmark.max_ticks_per_episode"
        )
        valid = set(AgentConfig.VALID_TYPES)
        kept = [a for a in self.agents if a in valid]
        if len(kept) != len(self.agents):
            log.warning("Dropping unknown benchmark agents: %s", set(self.agents) - valid)
        self.agents = kept or ["astar"]


@dataclass
class TrainingConfig:
    episodes: int = 400
    eval_episodes: int = 20
    eval_every_episodes: int = 50
    board_width: int = 20
    board_height: int = 15
    checkpoint_name: str = "dqn_default.pt"
    autosave_every_episodes: int = 50

    def validate(self) -> None:
        self.episodes = _clamp_int(self.episodes, 1, 100000, "training.episodes")
        self.eval_episodes = _clamp_int(self.eval_episodes, 1, 500, "training.eval_episodes")
        self.eval_every_episodes = _clamp_int(self.eval_every_episodes, 1, 10000, "training.eval_every_episodes")
        self.board_width = _clamp_int(self.board_width, 10, 40, "training.board_width")
        self.board_height = _clamp_int(self.board_height, 10, 30, "training.board_height")
        if not self.checkpoint_name.endswith(".pt"):
            self.checkpoint_name += ".pt"


@dataclass
class AppConfig:
    schema_version: int = _SCHEMA_VERSION
    board: BoardConfig = field(default_factory=BoardConfig)
    game: GameConfig = field(default_factory=GameConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        for section in (
            self.board,
            self.game,
            self.window,
            self.audio,
            self.visual,
            self.agent,
            self.llm,
            self.rl,
            self.safety,
            self.decision,
            self.benchmark,
            self.training,
        ):
            if hasattr(section, "validate"):
                section.validate()
        if int(self.schema_version) != _SCHEMA_VERSION:
            log.warning(
                "Config schema version %s != supported %s; loaded with defaults where needed.",
                self.schema_version,
                _SCHEMA_VERSION,
            )
            self.schema_version = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path else paths.default_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)  # atomic on POSIX and Windows
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        cfg_path = Path(path) if path else paths.default_config_path()
        cfg = cls()
        if not cfg_path.exists():
            log.info("No config file at %s; using defaults.", cfg_path)
            cfg.validate()
            return cfg
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be a JSON object")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            log.error("Could not read config %s (%s); using defaults.", cfg_path, exc)
            backup = cfg_path.with_suffix(cfg_path.suffix + ".corrupt")
            try:
                if cfg_path.exists():
                    cfg_path.replace(backup)
                    log.warning("Corrupt config moved to %s", backup)
            except OSError:
                pass
            cfg.validate()
            return cfg

        cfg._apply_dict(raw)
        cfg.validate()
        return cfg

    def _apply_dict(self, raw: dict[str, Any]) -> None:
        """Merge a (possibly partial/invalid) JSON dict into the dataclasses."""
        for section_name, section in (
            ("board", self.board),
            ("game", self.game),
            ("window", self.window),
            ("audio", self.audio),
            ("visual", self.visual),
            ("agent", self.agent),
            ("llm", self.llm),
            ("rl", self.rl),
            ("safety", self.safety),
            ("decision", self.decision),
            ("benchmark", self.benchmark),
            ("training", self.training),
        ):
            value = raw.get(section_name)
            if not isinstance(value, dict):
                continue
            for key, val in value.items():
                if not hasattr(section, key):
                    continue
                current = getattr(section, key)
                try:
                    if isinstance(current, bool) or isinstance(val, bool):
                        setattr(section, key, bool(val))
                    elif isinstance(current, int) and not isinstance(current, bool):
                        setattr(section, key, int(val))
                    elif isinstance(current, float):
                        setattr(section, key, float(val))
                    elif isinstance(current, list):
                        if section_name == "rl" and key == "hidden_sizes":
                            setattr(section, key, [int(h) for h in val][:3] if isinstance(val, list) else current)
                        elif section_name == "benchmark" and key == "agents":
                            setattr(section, key, [str(a) for a in val] if isinstance(val, list) else current)
                    elif isinstance(current, str):
                        setattr(section, key, str(val))
                    elif current is None:
                        setattr(section, key, val)
                except (TypeError, ValueError) as exc:
                    log.warning("Ignoring invalid config %s.%s=%r (%s)", section_name, key, val, exc)
            # Nested sections
            if section_name == "rl" and isinstance(value.get("reward"), dict):
                for key, val in value["reward"].items():
                    if hasattr(self.rl.reward, key):
                        try:
                            current = getattr(self.rl.reward, key)
                            if isinstance(current, int) and not isinstance(current, bool):
                                setattr(self.rl.reward, key, int(val))
                            elif isinstance(current, float):
                                setattr(self.rl.reward, key, float(val))
                        except (TypeError, ValueError):
                            pass
            if section_name == "decision":
                if isinstance(value.get("hybrid"), dict):
                    for key, val in value["hybrid"].items():
                        if hasattr(self.decision.hybrid, key):
                            try:
                                setattr(self.decision.hybrid, key, float(val))
                            except (TypeError, ValueError):
                                pass
                if isinstance(value.get("astar"), dict):
                    for key, val in value["astar"].items():
                        if hasattr(self.decision.astar, key):
                            try:
                                setattr(self.decision.astar, key, float(val))
                            except (TypeError, ValueError):
                                pass
        if "schema_version" in raw:
            try:
                self.schema_version = int(raw["schema_version"])
            except (TypeError, ValueError):
                pass


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load, validate, and return the application config."""
    cfg = AppConfig.load(path)
    # Environment-level overrides (documented in .env.example)
    if os.environ.get("SNAKE_ARENA_DISABLE_LLM", "0") == "1":
        cfg.llm.enabled = False
    host = os.environ.get("SNAKE_ARENA_OLLAMA_HOST", "").strip()
    if host:
        cfg.llm.host = host.rstrip("/")
    model = os.environ.get("SNAKE_ARENA_OLLAMA_MODEL", "").strip()
    if model:
        cfg.llm.model = model
    return cfg
