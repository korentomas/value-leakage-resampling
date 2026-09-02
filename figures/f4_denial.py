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
# Intervals are 95% equal-tailed posterior intervals (ETI), matching what
# swap/step4_labels.py stores in artifacts/step4_labels.json under
# contrast/Denies-Admits bet_shaped. The figure labels them ETI explicitly so
# the language cannot drift into "HDI" in a caption.
fig, (ax, axd) = plt.subplots(1, 2, figsize=(style.WIDE, 3.3),
                              gridspec_kw=dict(width_ratios=[2.6, 1.4], wspace=0.30))
style.trim(ax); style.trim(axd)
style.panel_label(ax, "(a)", y=1.06); style.panel_label(axd, "(b)", y=1.06)
xs = np.linspace(0.55, 1.0, 600)
BW = 0.6   # fractions are discrete (steps of 1/n); widen the kernel past one step

peak = max(gaussian_kde(frac[L], bw_method=BW)(xs).max() for L in ("Admits", "Denies"))
# (label x-anchor, label height as a multiple of the tallest density) — the two
# labels are staggered because at column width their bracketed numbers collide.
LAB = {"Denies": (0.700, 1.20), "Admits": (0.900, 1.00)}
for L, col in (("Admits", style.NEUTRAL), ("Denies", style.STEERED)):
    x = frac[L]
    y = gaussian_kde(x, bw_method=BW)(xs)
    ax.fill_between(xs, 0, y, color=col, alpha=0.18, lw=0)
    ax.plot(xs, y, color=col, lw=1.6)
    m, lo, hi = eti(x)
    ax.plot([lo, hi], [-0.06 * peak] * 2, color=col, lw=1.4, solid_capstyle="butt", clip_on=False)
    ax.plot([m], [-0.06 * peak], marker="o", ms=3.4, color=col, clip_on=False)
    lx, ly = LAB[L]
    ax.text(lx, ly * peak, f"{L}  (n = {n[L]})\n{m:.2f}  [{lo:.2f}, {hi:.2f}]",
            ha="center", va="bottom", fontsize=8, color=col, linespacing=1.3)

mM, loM, hiM = eti(frac["Mentions"])
ax.plot([mM, mM], [0, 0.10 * peak], color=style.INK, lw=0.9)
ax.text(mM, 0.12 * peak, f"Mentions\n(n = {n['Mentions']})  {mM:.2f}", ha="center", va="bottom",
        fontsize=7, color=style.INK, linespacing=1.3)

ax.set_xlim(0.6, 1.0)
ax.set_ylim(-0.13 * peak, peak * 1.52)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("share of the group whose prefix already leans to the good side\n(s > 0.2) or that a swapped prompt can still move")
ax.text(0.602, -0.06 * peak, "mean, 95% ETI", ha="left", va="center", fontsize=7,
        color=style.FAINT, clip_on=False)

# difference panel
md, lod, hid = eti(diff)
xd = np.linspace(-0.25, 0.05, 400)
kd = gaussian_kde(diff, bw_method=BW)(xd)
axd.fill_between(xd, 0, kd, color=style.STEERED, alpha=0.18, lw=0)
axd.plot(xd, kd, color=style.STEERED, lw=1.6)
axd.axvline(0, color=style.INK, lw=0.8, ls=(0, (3, 3)))
axd.plot([lod, hid], [-0.06 * kd.max()] * 2, color=style.STEERED, lw=1.4, clip_on=False)
axd.plot([md], [-0.06 * kd.max()], marker="o", ms=3.4, color=style.STEERED, clip_on=False)
axd.text(0.44, 0.88, f"{md:+.2f}  [{lod:+.2f}, {hid:+.2f}]\n95% ETI", transform=axd.transAxes,
         ha="center", va="bottom", fontsize=8, color=style.STEERED, linespacing=1.3)
axd.set_xlim(-0.25, 0.05)
axd.set_ylim(-0.13 * kd.max(), kd.max() * 1.34)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.set_xlabel("Denies \u2212 Admits")
axd.set_xticks([-0.2, -0.1, 0.0])

fig.suptitle("A chain of thought that declares an intention to ignore the bet leans less\ntoward the good side than one that admits the influence",
             fontsize=10, y=1.13)
style.save(fig, "F4_denial")

if __name__ == "__main__":
    for L in ("Admits", "Denies", "Mentions"):
        m, lo, hi = eti(frac[L]); print(f"{L:9s} n={n[L]:3d}  {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"Denies-Admits  {md:+.3f} [{lod:+.3f}, {hid:+.3f}]   P(diff<0)={np.mean(diff < 0):.4f}")
