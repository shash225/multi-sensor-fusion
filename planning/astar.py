"""Grid A* on a BEV cost map (8-connected, Euclidean heuristic)."""
from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_expansions: int = 60000,
          heuristic_weight: float = 1.0) -> tuple[list[tuple[int, int]], bool]:
    """Shortest path on a cost grid.

    Args:
        cost: ``(H, W)`` per-cell traversal cost (``>= 0``, ``np.inf`` = blocked). Moving into a
            cell costs ``step_length * (1 + cost[cell])``.
        start, goal: ``(row, col)`` cells.

    Returns:
        ``(path, reached)`` — if the goal is unreachable the path leads to the explored cell
        closest to the goal (best effort), and ``reached`` is ``False``.
    """
    h, w = cost.shape
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        raise ValueError("goal outside grid")
    g_score = np.full(cost.shape, np.inf)
    parent = -np.ones(cost.shape + (2,), dtype=np.int32)
    g_score[start] = 0.0

    def heuristic(r: int, c: int) -> float:
        return heuristic_weight * math.hypot(r - goal[0], c - goal[1])

    open_heap = [(heuristic(*start), 0.0, start)]
    best, best_h = start, heuristic(*start)
    closed = np.zeros(cost.shape, dtype=bool)
    expansions = 0
    while open_heap and expansions < max_expansions:
        _, g, node = heapq.heappop(open_heap)
        if closed[node]:
            continue
        closed[node] = True
        expansions += 1
        hn = heuristic(*node)
        if hn < best_h:
            best, best_h = node, hn
        if node == goal:
            best = goal
            break
        r, c = node
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or closed[nr, nc]:
                continue
            cell_cost = cost[nr, nc]
            if not np.isfinite(cell_cost):
                continue
            ng = g + step * (1.0 + cell_cost)
            if ng < g_score[nr, nc]:
                g_score[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(open_heap, (ng + heuristic(nr, nc), ng, (nr, nc)))
    path = [best]
    while path[-1] != start:
        pr, pc = parent[path[-1]]
        if pr < 0:
            break
        path.append((int(pr), int(pc)))
    path.reverse()
    return path, best == goal


def nearest_free(cost: np.ndarray, cell: tuple[int, int], max_radius: int = 20) -> Optional[tuple[int, int]]:
    """Closest finite-cost cell to ``cell`` within ``max_radius`` (square rings)."""
    h, w = cost.shape
    r0, c0 = cell
    if 0 <= r0 < h and 0 <= c0 < w and np.isfinite(cost[r0, c0]):
        return cell
    for rad in range(1, max_radius + 1):
        best, best_d = None, np.inf
        for r in range(r0 - rad, r0 + rad + 1):
            for c in (c0 - rad, c0 + rad) if abs(r - r0) != rad else range(c0 - rad, c0 + rad + 1):
                if 0 <= r < h and 0 <= c < w and np.isfinite(cost[r, c]):
                    d = math.hypot(r - r0, c - c0)
                    if d < best_d:
                        best, best_d = (r, c), d
        if best is not None:
            return best
    return None
