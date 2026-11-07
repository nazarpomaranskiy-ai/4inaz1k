"""Short-horizon collision forecast; coordinates are pixels, time is seconds."""
from dataclasses import dataclass
import math

@dataclass
class Shot:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

DIRECTIONS = [('STAY', 0, 0), ('W', 0, -1), ('WD', 1, -1),
              ('D', 1, 0), ('SD', 1, 1), ('S', 0, 1),
              ('SA', -1, 1), ('A', -1, 0), ('WA', -1, -1)]

def recommend(player, shots, arena, speed=180, radius=6, horizon=.4, margin=5):
    """Maximize worst predicted clearance, subject to rectangular arena bounds.

    Assumes constant shot velocity and instantaneous constant player velocity.
    Negative clearance means a predicted collision. No shots => no recommendation.
    """
    if not shots:
        return None, []
    px, py = player
    left, top, right, bottom = arena
    results = []
    for name, dx, dy in DIRECTIONS:
        norm = math.hypot(dx, dy) or 1
        vx, vy = speed * dx / norm, speed * dy / norm
        ex, ey = px + vx * horizon, py + vy * horizon
        if not all(left + radius <= x <= right - radius and
                   top + radius <= y <= bottom - radius
                   for x, y in [(px, py), (ex, ey)]):
            continue
        clearance = float('inf')
        for shot in shots:
            rx, ry = shot.x - px, shot.y - py
            ux, uy = shot.vx - vx, shot.vy - vy
            vv = ux * ux + uy * uy
            t = max(0, min(horizon, -(rx * ux + ry * uy) / vv)) if vv else 0
            gap = math.hypot(rx + ux*t, ry + uy*t) - radius - shot.radius - margin
            clearance = min(clearance, gap)
        results.append((name, clearance, vx, vy))
    return max(results, key=lambda r: r[1]) if results else None, results
