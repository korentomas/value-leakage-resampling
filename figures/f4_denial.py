"""F4 — posterior share of bet-shaped conversations by the paper's covertness
label (Admits / Denies / Mentions), neutral prior.

"Bet-shaped" per draw per unit = still reversible after t = 0.2
(step3_summaries dependent_after_t0) OR s(0.2) > 0.2, where s is the plug-in
neutral-minus-baseline shift from the (orig, neutral) fit, exactly as in
swap/step4_labels.py. Per-draw fraction within each label group; densities of
those fractions, plus the Denies - Admits difference.

Run:  ../.venv-model/bin/python f4_denial.py
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

import style  # noqa: E402

style.use()

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
fig, (ax, axd) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(width_ratios=[3, 1.35], wspace=0.28))
style.trim(ax); style.trim(axd)
xs = np.linspace(0.55, 1.0, 600)
BW = 0.6   # fractions are discrete (steps of 1/n); widen the kernel past one step

for L, col, dy in (("Admits", style.NEUTRAL, 1.0), ("Denies", style.STEERED, 1.0)):
    x = frac[L]
    kde = gaussian_kde(x, bw_method=BW)
    y = kde(xs)
    ax.fill_between(xs, 0, y, color=col, alpha=0.18, lw=0)
    ax.plot(xs, y, color=col, lw=2.0)
    m, lo, hi = eti(x)
    ax.plot([lo, hi], [-1.2, -1.2], color=col, lw=1.6, solid_capstyle="butt", clip_on=False)
    ax.plot([m], [-1.2], marker="o", ms=5, color=col, clip_on=False)
    ax.text(m, y.max() + 1.0, f"{L}  (n = {n[L]})\n{m:.2f}  [{lo:.2f}, {hi:.2f}]",
            ha="center", va="bottom", fontsize=9.5, color=col, linespacing=1.25)

mM, loM, hiM = eti(frac["Mentions"])
ax.plot([mM, mM], [0, 1.6], color=style.INK, lw=1.0)
ax.text(mM, 1.9, f"Mentions  (n = {n['Mentions']})\n{mM:.2f}", ha="center", va="bottom",
        fontsize=9, color=style.INK, linespacing=1.25)

ax.set_xlim(0.6, 1.0)
ax.set_ylim(-2.2, max(gaussian_kde(frac[L], bw_method=BW)(xs).max() for L in ("Admits", "Denies")) * 1.35)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("share of the group whose prefix leans to the good side (> 0.2) or is still reversible")
ax.text(0.6, -1.2, "mean, 95% ETI", ha="left", va="center", fontsize=8.5, color=style.FAINT, clip_on=False)

# difference panel
md, lod, hid = eti(diff)
xd = np.linspace(-0.25, 0.05, 400)
kd = gaussian_kde(diff, bw_method=BW)(xd)
axd.fill_between(xd, 0, kd, color=style.STEERED, alpha=0.18, lw=0)
axd.plot(xd, kd, color=style.STEERED, lw=2.0)
axd.axvline(0, color=style.INK, lw=0.9, ls=(0, (3, 3)))
axd.plot([lod, hid], [-0.05 * kd.max()] * 2, color=style.STEERED, lw=1.6, clip_on=False)
axd.plot([md], [-0.05 * kd.max()], marker="o", ms=5, color=style.STEERED, clip_on=False)
axd.text(md, kd.max() * 1.06, f"{md:+.2f}  [{lod:+.2f}, {hid:+.2f}]", ha="center", va="bottom",
         fontsize=9.5, color=style.STEERED)
axd.set_xlim(-0.25, 0.05)
axd.set_ylim(-0.12 * kd.max(), kd.max() * 1.35)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.set_xlabel("Denies − Admits")
axd.set_xticks([-0.2, -0.1, 0.0])

fig.suptitle("A chain of thought that declares an intention to ignore the bet leans less toward\nthe good side than one that admits the influence",
             fontsize=12.5, y=1.03)
style.save(fig, "F4_denial")

if __name__ == "__main__":
    for L in ("Admits", "Denies", "Mentions"):
        m, lo, hi = eti(frac[L]); print(f"{L:9s} n={n[L]:3d}  {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"Denies-Admits  {md:+.3f} [{lod:+.3f}, {hid:+.3f}]   P(diff<0)={np.mean(diff < 0):.4f}")
