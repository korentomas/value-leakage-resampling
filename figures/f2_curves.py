"""F2 — per-conversation dep(t) curves, grouped by whether the answer was
still reversible by a contradicting prompt after the first fifth of the CoT.

Data: artifacts/step3_summaries.json (neutral prior): per-unit dep_mean /
dep_lo95 / dep_hi95 on the grid, dependent_after_t0 flag.

Run:  ../external/value_leakage/.venv/bin/python f2_curves.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

import style

style.use()

S = json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"]
UNITS = [u for u in S if u["is_rollout"]]
grid = np.array(UNITS[0]["grid"])
xs = np.linspace(0, 1, 240)

late = [u for u in UNITS if u["dependent_after_t0"]]
early = [u for u in UNITS if not u["dependent_after_t0"]]

fig, ax = style.plate(figsize=(10, 6))

for group, faint, strong in ((early, style.FAINT, style.NEUTRAL), (late, style.FAINT_STEERED, style.STEERED)):
    for u in group:
        ax.plot(xs, PchipInterpolator(grid, u["dep_mean"])(xs), color=faint, lw=0.55, alpha=0.55, zorder=1)
    arr = np.array([u["dep_mean"] for u in group])
    m = arr.mean(axis=0)
    lo, hi = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)  # IQR across conversations
    ax.fill_between(xs, PchipInterpolator(grid, lo)(xs), PchipInterpolator(grid, hi)(xs),
                    color=strong, alpha=0.12, lw=0, zorder=2)
    ax.plot(xs, PchipInterpolator(grid, m)(xs), color=strong, lw=3.2, zorder=3)

pop0 = np.mean([u["dep_mean"][0] for u in UNITS])
ax.scatter([0], [pop0], color=style.INK, s=70, zorder=5)
style.annotate(ax, "published bias, 0.62", xy=(0, pop0), xytext=(0.08, 0.80))

ax.set_xlabel("t (fraction of CoT)")
ax.set_ylabel("dep(t)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.0)
ax.set_title("Most of the bet's hold on the answer is gone by t = 0.2", pad=14)
ax.legend(handles=[
    Line2D([], [], color=style.STEERED, lw=3.2, label=f"still reversible after t = 0.2  (n = {len(late)})"),
    Line2D([], [], color=style.NEUTRAL, lw=3.2, label=f"locked before t = 0.2  (n = {len(early)})"),
], loc="upper right")

for ext in ("png", "svg"):
    fig.savefig(f"out/{ext}/F2_final.{ext}", bbox_inches="tight", pad_inches=0.15, dpi=450 if ext == "png" else None)
print("saved out/svg/F2_final.svg + out/png/F2_final.png")
