"""F3 — three-arm population curves: share of continuations on the source's
favoured ("good") side, by cut, for the original / no-bet (neutral) / swapped
prompts, one panel per direction.

Model-free pooled rates (sum k / sum n), cuts binned to the grid 0,.2,...,1 by
round(frac_sentences*5)/5.
  orig, swap : artifacts/step3_counts.jsonl   (t=0 = is_cell_level rows with t==0)
  neutral    : artifacts/step4_neutral_counts.jsonl (no t=0 rows; its t=0 value is
               the no-bet baseline, artifacts/step4_baseline_counts.jsonl,
               unit-weighted per direction)

These are pooled point estimates, not posteriors — no band is drawn and the
caption says so. The per-conversation uncertainty lives in F2.

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
DIRS = [("above_good", "high number is the good one"),
        ("below_good", "low number is the good one")]
# (s label xy, dep label xy) in data coords, placed in regions verified empty
# for this data; s sits above dep where the s bracket is the higher one, so the
# two leader lines never cross.
LABEL_POS = {"above_good": ((0.42, 0.92), (0.46, 0.26)),
             "below_good": ((0.46, 0.13), (0.46, 0.34))}
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

fig, axes = plt.subplots(1, 2, figsize=(style.WIDE, 3.2), sharey=True)
for panel, (ax, (d, sub)) in zip("ab", zip(axes, DIRS)):
    style.trim(ax)
    style.panel_label(ax, f"({panel})", x=-0.015, y=1.10)
    b = baseline[d]
    ax.axhline(b, color=style.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, b + 0.02, "no-bet baseline", ha="right", va="bottom",
            fontsize=7, color=style.NEUTRAL, style="italic")
    for arm, lab, col in ARMS:
        ax.plot(GRID, rate[(d, arm)], color=col, lw=1.6, zorder=3,
                marker="o", ms=3.0, mec=style.BG, mew=0.6)
    # t = 0.2 gap annotations: thin double bracket at the cut, labels led out
    x = 0.2
    yo, yn, ys_ = (rate[(d, a)][1] for a in ("orig", "neutral", "swap"))
    for xb, lo, hi, col in ((x - 0.016, b, yn, style.NEUTRAL),
                            (x + 0.016, ys_, yo, style.STEERED)):
        ax.plot([xb, xb], [lo, hi], color=col, lw=0.8, zorder=2)
        for yy in (lo, hi):
            ax.plot([xb - 0.008, xb + 0.008], [yy, yy], color=col, lw=0.8, zorder=2)
    ts, td = LABEL_POS[d]
    style.annotate(ax, f"s = {yn - b:+.2f}", xy=(x - 0.016, (b + yn) / 2), xytext=ts,
                   color=style.NEUTRAL, fontsize=8, curve=-0.25)
    style.annotate(ax, f"dep = {yo - ys_:.2f}", xy=(x + 0.016, (ys_ + yo) / 2), xytext=td,
                   color=style.STEERED, fontsize=8, curve=0.25)

    ax.set_xticks(GRID)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{d.replace('_', '-')} — {sub}", fontsize=8.5, loc="left", pad=5)

axes[0].set_ylabel("share of continuations\non the good side")
fig.supxlabel("cut t (fraction of CoT kept as prefix)", fontsize=9, y=0.02)
fig.legend(handles=[Line2D([], [], color=c, lw=1.6, marker="o", ms=3.0, label=l)
                    for _, l, c in ARMS],
           loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8)
fig.suptitle("Once a prefix exists, the bet prompt adds nothing; the prefix carries the bias",
             fontsize=10, y=1.06)

style.save(fig, "F3_three_arm")

if __name__ == "__main__":
    for d, _ in DIRS:
        print(d, "baseline", round(baseline[d], 3))
        for arm, _, _ in ARMS:
            print(f"  {arm:8s}", np.round(rate[(d, arm)], 3))
