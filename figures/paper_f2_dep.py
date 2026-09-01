"""Paper F2 and F2b - dep(t) at the population level, and per conversation.

Two figures rather than one. Three panels plus a colourbar in a 5.5in text
width leaves each about 1.8in, and the heatmap is the panel that suffers, which
is backwards: it is the only display of the per-conversation labels the method
produces, and those are the contribution.

  F2_dep       population mean dep(t), and the distribution of lock positions
  F2b_perconv  one row per conversation, sorted by lock, full width

Run:  ../.venv-model/bin/python paper_f2_dep.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps

ps.use()

S = json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"]
U = [u for u in S if u["is_rollout"]]
grid = np.array(U[0]["grid"])
arr = np.array([u["dep_mean"] for u in U])
lock = np.array([u["commitment_frac"] for u in U])
order = np.argsort(lock)

# ---- F2: the population view -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

ax = axes[0]
m = arr.mean(axis=0)
lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
ax.fill_between(grid, lo, hi, color=ps.ORIG, alpha=0.2, lw=0, label="95% spread")
ax.plot(grid, m, color=ps.ORIG, marker="o", ms=4, lw=1.6, label="mean")
ax.axhline(0, color=ps.REF, ls="--", lw=0.9)
ax.set_xlabel("cut $t$")
ax.set_ylabel("dep($t$)")
ax.set_xticks(grid)
ax.set_ylim(-0.05, 1.0)
ax.set_title("(a) dependence on the bet")
ax.legend()

ax = axes[1]
ax.hist(lock, bins=np.linspace(0, 1, 21), color=ps.NEUTRAL, edgecolor="white", lw=0.5)
ax.axvline(np.median(lock), color=ps.REF, ls="--", lw=1.2,
           label=f"median {np.median(lock):.2f}")
ax.set_xlabel("lock position")
ax.set_ylabel("conversations")
ax.set_xlim(0, 1)
ax.set_title("(b) where the answer is settled")
ax.legend()

fig.tight_layout()
ps.save(fig, "F2_dep")

# ---- F2b: one row per conversation -----------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 2.9))
im = ax.imshow(arr[order], aspect="auto", origin="lower", cmap="viridis",
               vmin=0, vmax=1, interpolation="nearest",
               extent=[grid[0], grid[-1], 0, len(U)])
ax.set_xlabel("cut $t$")
ax.set_ylabel("conversations sorted by lock position")
ax.set_xticks(grid)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("dep($t$)")
fig.tight_layout()
ps.save(fig, "F2b_perconv")

if __name__ == "__main__":
    print("dep(t) mean", np.round(m, 3))
    print(f"lock median {np.median(lock):.3f}  by 0.2: {np.mean(lock <= 0.2):.3f}  "
          f"by 0.4: {np.mean(lock <= 0.4):.3f}  still dep >0.1 at t=1: {np.mean(arr[:,-1] > 0.1):.3f}")
