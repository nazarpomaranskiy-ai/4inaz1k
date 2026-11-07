"""Short-horizon collision forecast; coordinates are pixels, time is seconds."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Shot:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    ax: float = 0
    ay: float = 0
    uncertainty: float = 0

DIRECTIONS = [('STAY', 0, 0), ('W', 0, -1), ('WD', 1, -1),
              ('D', 1, 0), ('SD', 1, 1), ('S', 0, 1),
              ('SA', -1, 1), ('A', -1, 0), ('WA', -1, -1)]

def recommend(player, shots, arena, speed=180, radius=6, horizon=.4, margin=5):
    """Maximize worst predicted clearance, subject to rectangular arena bounds.

    Uses fitted shot acceleration and instantaneous constant player velocity.
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
        data=np.array([[s.x,s.y,s.vx,s.vy,s.radius,s.ax,s.ay,s.uncertainty] for s in shots])
        sx,sy,svx,svy,sr,sax,say,su=data.T
        steps=max(12,math.ceil(horizon/.015))
        times=np.linspace(0,horizon,steps+1)[:,None]
        rx=sx+svx*times+.5*sax*times**2-px-vx*times
        ry=sy+svy*times+.5*say*times**2-py-vy*times
        ux=np.diff(rx,axis=0);uy=np.diff(ry,axis=0)
        vv=ux*ux+uy*uy
        alpha=np.clip(np.divide(-(rx[:-1]*ux+ry[:-1]*uy),vv,
                                 out=np.zeros_like(vv),where=vv>0),0,1)
        distances=np.hypot(rx[:-1]+ux*alpha,ry[:-1]+uy*alpha)
        curve_margin=np.hypot(sax,say)*(horizon/steps)**2/8
        gaps=distances-radius-sr-margin-su*(1+times[1:])-curve_margin
        clearance=float(np.min(gaps))
        results.append((name, clearance, vx, vy))
    return max(results, key=lambda r: r[1]) if results else None, results
