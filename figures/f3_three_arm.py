"""F3 — three-arm population curves: share of continuations on the source's
favoured ("good") side, by cut, for the original / no-bet (neutral) / swapped
prompts, one panel per direction.

Model-free pooled rates (sum k / sum n), cuts binned to the grid 0,.2,...,1 by
round(frac_sentences*5)/5.
  orig, swap : artifacts/step3_counts.jsonl   (t=0 = is_cell_level rows with t==0)
  neutral    : artifacts/step4_neutral_counts.jsonl (no t=0 rows; its t=0 value is
               the no-bet baseline, artifacts/step4_baseline_counts.jsonl,
               unit-weighted per direction)

Run:  ../.venv-model/bin/python f3_three_arm.py
"""
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

style.use()

GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DIRS = [("above_good", "good cause needs a high number"),
        ("below_good", "good cause needs a low number")]
LABEL_POS = {"above_good": ((0.31, 0.655), (0.31, 0.565)),
             "below_good": ((0.33, 0.62), (0.04, 0.83))}
ARMS = [("orig", "original prompt", style.INK),
        ("neutral", "no-bet prompt", style.NEUTRAL),
        ("swap", "swapped prompt", style.STEERED)]


def load(p):
    return [json.loads(l) for l in open(p)]


step3 = load("../artifacts/step3_counts.jsonl")
neut = load("../artifacts/step4_neutral_counts.jsonl")
base = load("../artifacts/step4_baseline_counts.jsonl")

acc = defaultdict(lambda: [0, 0])
for r in step3 + neut:
    if r["is_cell_level"] and r["t"] != 0:
        continue
    tb = 0.0 if r["is_cell_level"] else round(r["frac_sentences"] * 5) / 5
    a = acc[(r["direction"], r["arm"], tb)]
    a[0] += r["k"]; a[1] += r["n"]

baseline = {}
for d, _ in DIRS:
    rs = [r for r in base if r["direction"] == d]
    baseline[d] = float(np.mean([r["k"] / r["n"] for r in rs]))

rate = {}
for d, _ in DIRS:
    for arm, _, _ in ARMS:
        ys = []
        for t in GRID:
            k, n = acc[(d, arm, t)]
            if n == 0:
                ys.append(baseline[d] if (arm == "neutral" and t == 0) else np.nan)
            else:
                ys.append(k / n)
        rate[(d, arm)] = np.array(ys)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, (d, sub) in zip(axes, DIRS):
    style.trim(ax)
    b = baseline[d]
    ax.axhline(b, color=style.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, b + 0.015, "no-bet baseline", ha="right", va="bottom",
            fontsize=9, color=style.NEUTRAL, style="italic")
    for arm, lab, col in ARMS:
        y = rate[(d, arm)]
        ax.plot(GRID, y, color=col, lw=2.4 if arm != "orig" else 2.0, zorder=3,
                marker="o", ms=4.5, mec=style.BG, mew=0.8)
    # t = 0.2 gap annotations: thin double bracket at the cut, labels led out
    x = 0.2
    yo, yn, ys = (rate[(d, a)][1] for a in ("orig", "neutral", "swap"))
    for xb, lo, hi, col in ((x - 0.014, b, yn, style.NEUTRAL), (x + 0.014, ys, yo, style.STEERED)):
        ax.plot([xb, xb], [lo, hi], color=col, lw=0.9, zorder=2)
        for yy in (lo, hi):
            ax.plot([xb - 0.007, xb + 0.007], [yy, yy], color=col, lw=0.9, zorder=2)
    ts, td = LABEL_POS[d]
    style.annotate(ax, f"s = {yn - b:+.2f}", xy=(x - 0.014, (b + yn) / 2), xytext=ts,
                   color=style.NEUTRAL, fontsize=9.5, curve=-0.2)
    style.annotate(ax, f"dep = {yo - ys:.2f}", xy=(x + 0.014, (ys + yo) / 2), xytext=td,
                   color=style.STEERED, fontsize=9.5, curve=0.2)

    ax.set_xticks(GRID)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("cut t (fraction of CoT kept as prefix)")
    ax.set_title(f"{d.replace('_', '-')}: {sub}", fontsize=11, loc="left", pad=8)

axes[0].set_ylabel("share of continuations on the good side")
axes[0].legend(handles=[Line2D([], [], color=c, lw=2.4, marker="o", ms=4.5, label=l)
                        for _, l, c in ARMS], loc="lower right", fontsize=9.5)
fig.suptitle("Once a prefix exists, the bet prompt adds nothing; the prefix carries the bias",
             fontsize=12.5, y=1.0)
fig.tight_layout()

style.save(fig, "F3_three_arm")

if __name__ == "__main__":
    for d, _ in DIRS:
        print(d, "baseline", round(baseline[d], 3))
        for arm, _, _ in ARMS:
            print(f"  {arm:8s}", np.round(rate[(d, arm)], 3))
