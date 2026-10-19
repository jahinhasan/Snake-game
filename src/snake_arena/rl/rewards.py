"""Reward design for the DQN agent.

Per-step reward (all weights configurable in config/defaults rl.reward):

    + food_scale * value           on eating a positive food
                                   (normal +1, bonus +2.5, golden +5, special +10)
    + trap_penalty                 on eating trap food (default -2)
    + progress_scale * (d_prev - d_new)
                                   potential-style shaping toward the current
                                   target food (Ng et al. 1999): preserves the
                                   optimal policy while speeding learning
    - stagnation_penalty * over    once steps_since_food exceeds
                                   stagnation_start_steps; over grows slowly,
                                   punishing endless circling without progress
    + step_cost                    (default 0; available for tuning)
    - death_wall / death_self / death_opponent   on terminal collision

Terminal deaths dominate any accumulated shaping (10 vs <= ~1 per episode of
shaping), so the agent cannot profit by surviving safely forever without
eating: hunger stagnation and shaping toward food push it to actually score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import RLRewardConfig
from ..game.environment import StateView
from ..game.types import manhattan

if TYPE_CHECKING:
    from ..game.engine import GameEvent


@dataclass
class StepReward:
    total: float
    components: dict[str, float]


class RewardCalculator:
    """Turns raw engine events into the DQN reward signal."""

    def __init__(self, cfg: RLRewardConfig, width: int, height: int) -> None:
        self.cfg = cfg
        self.norm = max(1, width + height)

    @staticmethod
    def target_distance(view: StateView) -> float | None:
        if view.target is None:
            return None
        return float(manhattan(view.target.pos, view.me.head))

    def compute(
        self,
        events: list["GameEvent"],
        prev_dist: float | None,
        new_dist: float | None,
        death_reason: str | None,
        steps_since_food: int,
        terminated: bool,
    ) -> StepReward:
        cfg = self.cfg
        components: dict[str, float] = {}
        total = cfg.step_cost
        if cfg.step_cost:
            components["step"] = cfg.step_cost

        for event in events:
            if event.kind == "eat":
                gained = cfg.food_scale * event.value
                components["food"] = gained
                total += gained
            elif event.kind == "trap":
                components["trap"] = cfg.trap_penalty
                total += cfg.trap_penalty
            elif event.kind == "death":
                key = {
                    "wall": "death_wall",
                    "self": "death_self",
                    "head_on": "death_opponent",
                    "opponent": "death_opponent",
                }.get(event.reason or "", "death_self")
                penalty = getattr(cfg, key, cfg.death_self)
                components["death"] = penalty
                total += penalty

        if not terminated and prev_dist is not None and new_dist is not None:
            progress = (prev_dist - new_dist) * cfg.progress_scale
            components["progress"] = progress
            total += progress

        start = cfg.stagnation_start_steps
        if steps_since_food > start:
            over = (steps_since_food - start) / 10.0
            penalty = -cfg.stagnation_penalty * min(over, 20.0)
            components["stagnation"] = penalty
            total += penalty

        return StepReward(total=total, components=components)
