"""A* and BFS pathfinding on constructed grids."""

from __future__ import annotations

from snake_arena.agents.astar import astar_search, route_is_survivable
from snake_arena.agents.bfs import bfs_search
from snake_arena.game.types import manhattan


def _is_valid_path(path, start, goal, blocked):
    assert path[0] == start
    assert path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1, f"non-adjacent step {a}->{b}"
        assert b not in blocked, f"path enters blocked cell {b}"


def test_astar_straight_line_is_optimal():
    path = astar_search((1, 1), (5, 1), 10, 10, set())
    assert path is not None
    assert len(path) == 5  # manhattan distance 4 + start node
    _is_valid_path(path, (1, 1), (5, 1), set())


def test_astar_detours_around_wall():
    blocked = {(2, 1), (2, 2), (2, 3), (2, 4)}
    path = astar_search((1, 3), (4, 3), 10, 10, blocked)
    assert path is not None
    _is_valid_path(path, (1, 3), (4, 3), blocked)
    assert len(path) > 5  # detour is longer than the direct route


def test_astar_unreachable_returns_none():
    # Goal fully walled in.
    goal = (5, 5)
    blocked = {(4, 5), (6, 5), (5, 4), (5, 6)}
    assert astar_search((1, 1), goal, 10, 10, blocked) is None


def test_astar_start_equals_goal():
    assert astar_search((3, 3), (3, 3), 10, 10, set()) == [(3, 3)]


def test_bfs_matches_astar_length_on_open_grid():
    for goal in [(8, 3), (0, 9), (9, 9)]:
        a = astar_search((1, 1), goal, 10, 10, set())
        b = bfs_search((1, 1), goal, 10, 10, set())
        assert a is not None and b is not None
        assert len(a) == len(b), f"length mismatch for goal {goal}"
        _is_valid_path(b, (1, 1), goal, set())


def test_bfs_unreachable_returns_none():
    blocked = {(4, 5), (6, 5), (5, 4), (5, 6)}
    assert bfs_search((1, 1), (5, 5), 10, 10, blocked) is None


def test_route_survivability_open_area():
    path = astar_search((1, 1), (5, 1), 20, 20, set())
    assert path is not None
    assert route_is_survivable(path, set(), 20, 20, snake_length=5)


def test_route_into_tiny_pocket_is_not_survivable():
    # A wall splits a 6x3 board; the left half is a 9-cell pocket.
    goal = (2, 1)
    wall = {(3, 0), (3, 1), (3, 2)}
    path = astar_search((0, 1), goal, 6, 3, wall)
    assert path is not None  # the goal is reachable...
    # Flood from the path end reaches 8 cells (the path interior is occupied).
    assert route_is_survivable(path, wall, 6, 3, snake_length=8) is True
    assert route_is_survivable(path, wall, 6, 3, snake_length=9) is False
