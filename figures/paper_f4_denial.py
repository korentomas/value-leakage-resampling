"""Paper F4 - bet-shaped share by the paper's covertness label.

Replaces the blog figure's kernel densities and floating text labels with a bar
chart carrying 95% intervals, which is the idiom Betley et al. use for this kind
of group comparison, plus a second panel for the Denies - Admits contrast.

Run:  ../.venv-model/bin/python paper_f4_denial.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, "../swap")
from model import DEFAULT_GRID, load_counts, prepare_data  # noqa: E402
from three_arm import PAIRS, merge_records, shared_baseline_s_draws  # noqa: E402

import paper_style as ps  # noqa: E402

ps.use()

A = Path("../artifacts")
PRIOR, EPS, T = "neutral", 0.2, 0.2

step3 = load_counts(A / "step3_counts.jsonl")
neutral = load_counts(A / "step4_neutral_counts.jsonl")
baseline = load_counts(A / "step4_baseline_counts.jsonl")
merged, _ = merge_records(step3, neutral, baseline)
cbc = {r["cell_id"]: (int(r["k"]), int(r["n"])) for r in baseline}
arm_a, arm_b = PAIRS["r"]
data = prepare_data(merged, grid=DEFAULT_GRID, arm_a=arm_a, arm_b=arm_b)
g = int(np.argmin(np.abs(np.asarray(DEFAULT_GRID) - T)))

st3 = {u["unit_id"]: u for u in json.load(open(A / "step3_summaries.json"))[PRIOR]["summaries"] if u["is_rollout"]}
lab = {}
for l in open(A / "prefix_direction.jsonl"):
    r = json.loads(l); lab[r["source_id"]] = r["denial_label"]

idata = az.from_netcdf(A / f"step4_fits/{PRIOR}/step4_idata_r_{PRIOR}.nc")
s = shared_baseline_s_draws(idata, data, cbc, pair="r")[:, :, g]          # (draws, units)
units = list(data.unit_ids)
roll = [i for i, u in enumerate(units) if u in st3]
late = np.array([st3[units[i]]["dependent_after_t0"] for i in roll])
shaped = (s[:, roll] > EPS) | late[None, :]                                # (draws, rollouts)
labels = np.array([lab.get(units[i]) for i in roll])

frac = {L: shaped[:, labels == L].mean(axis=1) for L in ("Admits", "Denies", "Mentions")}
n = {L: int((labels == L).sum()) for L in frac}
diff = frac["Denies"] - frac["Admits"]


def eti(x):
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))



# ---- figure ----------------------------------------------------------------
# Single column. Two panels did not fit at 3.4 in without the group labels
# colliding, so the contrast is annotated in the axes instead of given a panel.
LABELS = ["Admits", "Denies", "Mentions"]
means, los, his = [], [], []
for L in LABELS:
    m, lo, hi = eti(frac[L])
    means.append(m); los.append(m - lo); his.append(hi - m)
md, lod, hid = eti(diff)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = np.arange(len(LABELS))
ax.bar(xs, means, yerr=[los, his], capsize=3, width=0.65,
       color=[ps.ORIG, ps.SWAP, ps.NEUTRAL], edgecolor="black", linewidth=0.5)
ax.set_xticks(xs)
ax.set_xticklabels([f"{L}\nn = {n[L]}" for L in LABELS])
ax.set_ylabel("bet-shaped share")
ax.set_ylim(0, 1.18)
ax.text(0.5, 0.97, f"Denies $-$ Admits  ${md:+.2f}$ $[{lod:+.2f}, {hid:+.2f}]$",
        transform=ax.transAxes, ha="center", va="top", fontsize=ps.ANNOT_FS)
fig.tight_layout()
ps.save(fig, "F4_denial")

if __name__ == "__main__":
    for L in LABELS:
        m, lo, hi = eti(frac[L]); print(f"{L:9s} n={n[L]:3d}  {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"Denies-Admits  {md:+.3f} [{lod:+.3f}, {hid:+.3f}]")
