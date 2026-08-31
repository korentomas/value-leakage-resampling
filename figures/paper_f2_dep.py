"""Paper F2 - dep(t): population trend, every conversation, and the lock distribution.

Three panels, each doing work the others cannot:

  (a) mean dep(t) with a 95% spread, which carries the headline numbers
      (0.62 at t=0, 0.26 at t=0.2).
  (b) a heatmap with one ROW PER CONVERSATION, sorted by lock. This is the only
      panel in the paper that shows the per-conversation labels the method
      produces, which is the contribution; the mean in (a) hides the fact that a
      minority of conversations stay dependent on the bet out to t = 1.
  (c) the distribution of lock positions, which is (b) summarised.

The heatmap is deliberate rather than decorative: with 250 units it is the only
readable way to show every label at once. The other paper figures compare two or
three groups, where bars and lines are better, so none of them use colour maps.

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

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5),
                         gridspec_kw=dict(width_ratios=[1.05, 1.0, 1.05]))

# (a) population trend
ax = axes[0]
m = arr.mean(axis=0)
lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
ax.fill_between(grid, lo, hi, color=ps.ORIG, alpha=0.2, lw=0, label="95% spread")
ax.plot(grid, m, color=ps.ORIG, marker="o", ms=3.5, lw=1.5, label="mean")
ax.axhline(0, color=ps.REF, ls="--", lw=0.9)
ax.set_xlabel("cut $t$")
ax.set_ylabel("dep($t$)")
ax.set_xticks(grid[::2])
ax.set_ylim(-0.05, 1.0)
ax.set_title("(a) population")
ax.legend(loc="upper right")

# (b) one row per conversation
ax = axes[1]
im = ax.imshow(arr[order], aspect="auto", origin="lower", cmap="viridis",
               vmin=0, vmax=1, interpolation="nearest",
               extent=[grid[0], grid[-1], 0, len(U)])
ax.set_xlabel("cut $t$")
ax.set_ylabel("conversations, sorted by lock")
ax.set_xticks(grid[::2])
ax.set_title("(b) every conversation")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("dep($t$)", fontsize=ps.ANNOT_FS)
cb.ax.tick_params(labelsize=ps.TICK_FS - 1)

# (c) lock distribution
ax = axes[2]
ax.hist(lock, bins=np.linspace(0, 1, 21), color=ps.NEUTRAL, edgecolor="white", lw=0.5)
ax.axvline(np.median(lock), color=ps.REF, ls="--", lw=1.2,
           label=f"median {np.median(lock):.2f}")
ax.set_xlabel("lock position")
ax.set_ylabel("conversations")
ax.set_xlim(0, 1)
ax.set_title("(c) lock distribution")
ax.legend(loc="upper right")

fig.tight_layout()
ps.save(fig, "F2_dep")

if __name__ == "__main__":
    print("dep(t) mean", np.round(m, 3))
    print(f"lock median {np.median(lock):.3f}  by 0.2: {np.mean(lock <= 0.2):.3f}  "
          f"by 0.4: {np.mean(lock <= 0.4):.3f}  still dep at t=1 (>0.1): {np.mean(arr[:,-1] > 0.1):.3f}")
