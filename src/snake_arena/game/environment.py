"""Observation building and the single-snake RL environment.

``StateView`` precomputes everything agents need each tick (blocked cells,
ranked food priorities) so decision code stays cheap and consistent.
``ObservationBuilder.build`` produces the documented feature vector shared
verbatim by RL training and inference.

Observation layout (OBS_SIZE = 30 floats, all roughly in [0, 1]):

    [0:3]   immediate danger straight/left/right (wall, body, or opponent)
    [3:6]   reachable free area after moving straight/left/right,
            min(1, area / snake_length) -- dead-end signal
    [6:10]  one-hot current heading (UP, DOWN, LEFT, RIGHT)
    [10:14] target food signs: dx>0, dx<0, dy>0, dy<0
    [14:15] target food manhattan/(w+h)
    [15:16] target food |value|/100
    [16:17] target food is_trap
    [17:21] second-ranked food signs: dx>0, dx<0, dy>0, dy<0
    [21:22] opponent present
    [22:26] opponent head signs: dx>0, dx<0, dy>0, dy<0
    [26:27] opponent head manhattan/(w+h)
    [27:28] opponent is longer than us
    [28:29] snake length / board area (capped at 1)
    [29:30] hunger: min(1, steps_since_food / 150)

The target food is the best non-trap item by value/(1+manhattan); if only
traps remain, the nearest item of any kind is used so the agent still sees a
sensible direction signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .collision import flood_fill_count, is_deadly_for, planning_blocked_cells
from .engine import GameEngine, SnakeSpec  # noqa: F401  (re-exported for convenience)
from .food import Food, FoodCatalog
from .types import DIR_VECTORS, Action, Cell, manhattan

if TYPE_CHECKING:
    from .snake import Snake
    from .state import GameState

OBS_SIZE = 30

# Turn tables for relative frames.
_LEFT_OF: dict[Action, Action] = {
    Action.UP: Action.LEFT,
    Action.LEFT: Action.DOWN,
    Action.DOWN: Action.RIGHT,
    Action.RIGHT: Action.UP,
}
_RIGHT_OF: dict[Action, Action] = {v: k for k, v in _LEFT_OF.items()}


@dataclass
class StateView:
    """Per-tick cached decision context for one snake."""

    state: "GameState"
    snake_id: str
    me: "Snake"
    opponents: list["Snake"]
    blocked: set[Cell] = field(default_factory=set)  # tail-aware planning walls
    foods_sorted: list[Food] = field(default_factory=list)
    danger: set[Cell] = field(default_factory=set)
    target: Food | None = None
    target_secondary: Food | None = None

    @property
    def width(self) -> int:
        return self.state.width

    @property
    def height(self) -> int:
        return self.state.height


def food_priority(food: Food, head: Cell) -> float:
    """Default desirability heuristic: reward over travel cost; traps last."""
    if food.value < 0:
        return -1e6
    return food.value / (1.0 + manhattan(food.pos, head))


def build_state_view(state: "GameState", snake_id: str) -> StateView:
    me = state.snakes[snake_id]
    head = me.head
    view = StateView(
        state=state,
        snake_id=snake_id,
        me=me,
        opponents=state.opponents_of(snake_id),
        blocked=planning_blocked_cells(state, snake_id),
    )
    ranked = sorted(state.foods, key=lambda f: food_priority(f, head), reverse=True)
    view.foods_sorted = ranked
    positives = [f for f in ranked if f.value > 0]
    if positives:
        view.target = positives[0]
        view.target_secondary = positives[1] if len(positives) > 1 else None
    elif ranked:  # only traps left: nearest item as a direction signal
        view.target = min(ranked, key=lambda f: manhattan(f.pos, head))
        view.target_secondary = None
    return view


def _sign_feats(feats: np.ndarray, offset: int, dx: int, dy: int) -> None:
    feats[offset] = 1.0 if dx > 0 else 0.0
    feats[offset + 1] = 1.0 if dx < 0 else 0.0
    feats[offset + 2] = 1.0 if dy > 0 else 0.0
    feats[offset + 3] = 1.0 if dy < 0 else 0.0


class ObservationBuilder:
    """Builds the fixed-size feature vector documented in the module header."""

    def __init__(self, flood_limit: int = 900) -> None:
        self.flood_limit = flood_limit

    def build(self, state: "GameState", snake_id: str, view: StateView | None = None) -> np.ndarray:
        view = view or build_state_view(state, snake_id)
        me: Snake = view.me
        head = me.head
        feats = np.zeros(OBS_SIZE, dtype=np.float32)

        # Danger + free-space for straight/left/right relative to heading.
        for i, action in enumerate((me.direction, _LEFT_OF[me.direction], _RIGHT_OF[me.direction])):
            dx, dy = DIR_VECTORS[action]
            cell = (head[0] + dx, head[1] + dy)
            if is_deadly_for(state, me, cell):
                feats[i] = 1.0
            else:
                area = flood_fill_count(cell, view.blocked, state.width, state.height, self.flood_limit)
                feats[3 + i] = min(1.0, area / max(1, me.length))

        feats[6 + int(me.direction)] = 1.0

        if view.target is not None:
            tx, ty = view.target.pos
            _sign_feats(feats, 10, tx - head[0], ty - head[1])
            feats[14] = min(1.0, manhattan(view.target.pos, head) / (state.width + state.height))
            feats[15] = min(1.0, abs(view.target.value) / 100.0)
            feats[16] = 1.0 if view.target.value < 0 else 0.0
        if view.target_secondary is not None:
            sx, sy = view.target_secondary.pos
            _sign_feats(feats, 17, sx - head[0], sy - head[1])

        feats[21] = 1.0 if view.opponents else 0.0
        if view.opponents:
            opp = min(view.opponents, key=lambda o: manhattan(o.head, head))
            ox, oy = opp.head
            _sign_feats(feats, 22, ox - head[0], oy - head[1])
            feats[26] = min(1.0, manhattan(opp.head, head) / (state.width + state.height))
            feats[27] = 1.0 if opp.length > me.length else 0.0

        feats[28] = min(1.0, me.length / max(1, state.board_area))
        feats[29] = min(1.0, me.steps_since_food / 150.0)
        return feats


class SnakeEnv:
    """Single-snake environment wrapping the authoritative engine.

    Returns raw game events; reward shaping lives in ``rl/rewards.py`` so the
    game layer never depends on the RL layer. Trainer usage::

        obs = env.reset(seed)
        obs, events, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 15,
        moves_per_second: int = 0,
        max_foods: int = 4,
        seed: int | None = None,
        max_steps: int = 1200,
        catalog: FoodCatalog | None = None,
        time_limit_seconds: int = 300,
    ) -> None:
        self.max_steps = max_steps
        self._step_count = 0
        self.engine = GameEngine(
            width=width,
            height=height,
            moves_per_second=max(1, moves_per_second),
            max_foods=max_foods,
            seed=seed,
            time_limit_seconds=time_limit_seconds,
            catalog=catalog,
        )
        self.obs_builder = ObservationBuilder()
        self.snake_id = self.engine.specs[0].id
        self._last_view: StateView | None = None

    @property
    def state(self) -> "GameState":
        return self.engine.state

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.engine.reset(seed)
        self._step_count = 0
        self._last_view = build_state_view(self.engine.state, self.snake_id)
        return self.obs_builder.build(self.engine.state, self.snake_id, self._last_view)

    def step(self, action: int | Action):
        act = Action(int(action))
        result = self.engine.step({self.snake_id: act})
        self._step_count += 1
        self._last_view = build_state_view(self.engine.state, self.snake_id)
        obs = self.obs_builder.build(self.engine.state, self.snake_id, self._last_view)
        terminated = result.done
        truncated = not result.done and self._step_count >= self.max_steps
        info = {
            "events": result.events,
            "score": self.engine.state.snakes[self.snake_id].score,
            "length": self.engine.state.snakes[self.snake_id].length,
            "steps": self._step_count,
            "view": self._last_view,
            "engine_result": result,
        }
        return obs, result.events, terminated, truncated, info
