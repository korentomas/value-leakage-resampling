"""Paper F3 - share on the favoured side by cut, for the three prompt conditions.

Same data as the blog F3. Differences: Wilson 95% intervals are drawn instead of
bracket-and-arrow callouts, and s and dep move into the caption.

Run:  ../.venv-model/bin/python paper_f3_arms.py
"""
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps

ps.use()

GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DIRS = [("above_good", "above-good"), ("below_good", "below-good")]
ARMS = [("orig", "original", ps.ORIG), ("neutral", "no-bet", ps.NEUTRAL),
        ("swap", "swapped", ps.SWAP)]


def load(p):
    return [json.loads(l) for l in open(p)]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


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

fig, axes = plt.subplots(1, 2, figsize=ps.WIDE, sharey=True)
for ax, (d, sub) in zip(axes, DIRS):
    ax.axhline(baseline[d], color=ps.REF, ls="--", lw=0.9)
    for arm, lab, col in ARMS:
        ys, los, his = [], [], []
        for t in GRID:
            k, n = acc[(d, arm, t)]
            if n == 0 and arm == "neutral" and t == 0:
                p = baseline[d]; l = h = np.nan
            else:
                p, l, h = wilson(k, n)
            ys.append(p); los.append(p - l if l == l else 0); his.append(h - p if h == h else 0)
        ax.errorbar(GRID, ys, yerr=[los, his], color=col, marker="o", ms=4,
                    lw=1.6, capsize=2, label=lab)
    ax.set_xticks(GRID)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("cut $t$")
    ax.set_title(f"({'a' if d.startswith('above') else 'b'}) {sub}")
axes[0].set_ylabel("share on the favoured side")
fig.tight_layout()
ps.below_legend(fig, axes[0], ncol=3, y=0.02, title="prompt condition")
ps.save(fig, "F3_arms")
