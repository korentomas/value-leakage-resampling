"""F2 — per-conversation dep(t) curves, grouped by whether the answer was
still reversible by a contradicting prompt after the first fifth of the CoT.

Data: artifacts/step3_summaries.json (neutral prior): per-unit dep_mean /
dep_lo95 / dep_hi95 on the grid, dependent_after_t0 flag.

Bands are the interquartile range ACROSS CONVERSATIONS (not a posterior
interval) — the caption must say so; see figure-texts.md.

Curves are drawn with a shape-preserving (Pchip) interpolation through the 5
grid points purely for legibility. Pchip can still undershoot between knots —
24 of 250 units dip below 0, worst -0.16 — so every interpolated series is
clipped to [0, 1], the range dep is defined on.

Run:  ../.venv-model/bin/python f2_curves.py
"""
import json
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

import style

style.use()

S = json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"]
UNITS = [u for u in S if u["is_rollout"]]
grid = np.array(UNITS[0]["grid"])
xs = np.linspace(0, 1, 240)


def smooth(y):
    """Pchip through the grid knots, clipped to dep's valid range [0, 1]."""
    return np.clip(PchipInterpolator(grid, y)(xs), 0.0, 1.0)


late = [u for u in UNITS if u["dependent_after_t0"]]
early = [u for u in UNITS if not u["dependent_after_t0"]]

fig, ax = style.plate(figsize=(style.WIDE, 4.0))

for group, faint, strong in ((early, style.FAINT, style.NEUTRAL),
                             (late, style.FAINT_STEERED, style.STEERED)):
    for u in group:
        ax.plot(xs, smooth(u["dep_mean"]), color=faint, lw=0.35, alpha=0.32, zorder=1)
    arr = np.array([u["dep_mean"] for u in group])
    lo, hi = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)  # IQR across conversations
    ax.fill_between(xs, smooth(lo), smooth(hi), color=strong, alpha=0.22, lw=0, zorder=2)
    ax.plot(xs, smooth(arr.mean(axis=0)), color=strong, lw=2.4, zorder=3)

pop0 = float(np.mean([u["dep_mean"][0] for u in UNITS]))
ax.scatter([0], [pop0], color=style.INK, s=34, zorder=5)
style.annotate(ax, f"published bias, {pop0:.2f}", xy=(0, pop0), xytext=(0.10, 0.83))

ax.set_xlabel("t (fraction of the chain of thought kept as prefix)")
ax.set_ylabel("dep(t)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.0)
ax.set_xticks(grid)
ax.set_title("Most of the bet's hold on the answer is gone by t = 0.2", pad=10)
ax.legend(handles=[
    Line2D([], [], color=style.STEERED, lw=2.4, label=f"still reversible after t = 0.2  (n = {len(late)})"),
    Line2D([], [], color=style.NEUTRAL, lw=2.4, label=f"locked before t = 0.2  (n = {len(early)})"),
], loc="upper right")

style.save(fig, "F2_final")

if __name__ == "__main__":
    arr = np.array([u["dep_mean"] for u in UNITS])
    print("n units", len(UNITS), "late", len(late), "early", len(early))
    print("population dep(t) mean", np.round(arr.mean(axis=0), 3), "on grid", grid.tolist())
