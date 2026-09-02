"""F5 — when the claims are written, as fractions of the chain of thought.

Top: the 77 denying conversations in which the honesty claim could be located
(artifacts/denial_positions.jsonl, joined on step3 unit_id): lock position
(posterior median commitment_frac, neutral prior, step3_summaries) vs the first
honesty claim, one dumbbell per conversation sorted by lock.

Bottom: the 147 conversations with an explicit statement of intent to fall on
the good side (artifacts/prefix_direction.jsonl, grid_idx == 5, has_intent):
lock position and the intent statement's position, as jittered strips.

Run:  ../.venv-model/bin/python f5_timing.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

style.use()

S = {u["unit_id"]: u for u in json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
den = [json.loads(l) for l in open("../artifacts/denial_positions.jsonl")]
den = [r for r in den if r["rollout_id"] in S]
pdr = [json.loads(l) for l in open("../artifacts/prefix_direction.jsonl")]
intent = [r for r in pdr if r["grid_idx"] == 5 and r["has_intent"] and r["intent_pos_full"] is not None]

lock_d = np.array([S[r["rollout_id"]]["commitment_frac"] for r in den])
first = np.array([r["first_frac"] for r in den])
last = np.array([r["last_frac"] for r in den])
lock_i = np.array([S[r["source_id"]]["commitment_frac"] for r in intent])
ipos = np.array([r["intent_pos_full"] for r in intent])

order = np.argsort(lock_d)
lock_d, first, last = lock_d[order], first[order], last[order]
after = first > lock_d

C_LOCK, C_CLAIM, C_INTENT = style.INK, style.NEUTRAL, style.STEERED

fig, (ax, axb) = plt.subplots(2, 1, figsize=(style.WIDE, 6.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[4.2, 1.25], hspace=0.12))
style.trim(ax); style.trim(axb)
style.panel_label(ax, "(a)", y=1.14); style.panel_label(axb, "(b)", y=1.02)

# ---- top: dumbbells ----------------------------------------------------------
n = len(den)
ys = np.arange(n)
for i in range(n):
    ax.plot([lock_d[i], first[i]], [ys[i], ys[i]], color=style.FAINT_STEERED if after[i] else style.FAINT,
            lw=0.8, zorder=1, solid_capstyle="round")
ax.scatter(first, ys, s=7, color=C_CLAIM, zorder=3, lw=0)
ax.scatter(lock_d, ys, s=7, color=C_LOCK, zorder=4, lw=0)

for x, col, lab, dy in ((np.median(lock_d), C_LOCK, "lock", 2.2), (np.median(first), C_CLAIM, "first claim", 0.2)):
    ax.plot([x, x], [-1.5, n + 0.5], color=col, lw=0.7, ls=(0, (3, 3)), zorder=2)
ax.text(np.median(lock_d) + 0.012, n + 1.2, f"median lock {np.median(lock_d):.2f}", color=C_LOCK, fontsize=7.5, ha="left", va="bottom")
ax.text(np.median(first) - 0.012, n + 1.2, f"median first claim {np.median(first):.2f}", color=C_CLAIM, fontsize=7.5, ha="right", va="bottom")

ax.set_ylim(-1.5, n + 7)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_ylabel(f"{n} denying conversations, sorted by lock", labelpad=4, fontsize=8.5)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_LOCK, ms=3.4, label="answer locked (posterior median)"),
    Line2D([], [], marker="o", ls="", color=C_CLAIM, ms=3.4, label="first honesty claim"),
    Line2D([], [], color=style.FAINT_STEERED, lw=1.4, label=f"claim written after the lock  ({after.mean():.0%})"),
    Line2D([], [], color=style.FAINT, lw=1.4, label=f"claim written before the lock  ({1 - after.mean():.0%})"),
], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8, columnspacing=1.4)

# ---- bottom: intent strips -----------------------------------------------------
rng = np.random.default_rng(0)
m = len(intent)
y_lock, y_int = 1.0, 0.0
ax_j = 0.16
axb.scatter(lock_i, y_lock + rng.uniform(-ax_j, ax_j, m), s=6, color=C_LOCK, alpha=0.7, lw=0, zorder=3)
axb.scatter(ipos, y_int + rng.uniform(-ax_j, ax_j, m), s=6, color=C_INTENT, alpha=0.75, lw=0, zorder=3)
for y, x, col in ((y_lock, np.median(lock_i), C_LOCK), (y_int, np.median(ipos), C_INTENT)):
    axb.plot([x, x], [y - 0.3, y + 0.3], color=col, lw=1.8, zorder=4)
    axb.text(x, y + 0.34, f"{x:.2f}", ha="center", va="bottom", fontsize=7.5, color=col)
axb.set_yticks([y_lock, y_int])
style.no_minor(axb)
axb.set_yticklabels(["lock", "first intent\nstatement"], fontsize=8)
axb.tick_params(axis="y", length=0)
axb.spines["left"].set_visible(False)
axb.set_ylim(-0.55, 2.05)
# placed above the top strip: at column width this text lands on the lock points
# if it is anchored inside the plotted rows.
axb.text(0.0, 2.0, f"{m} conversations with an explicit intent to fall on the good side; "
                   f"intent written after the lock in {np.mean(ipos > lock_i):.0%}",
         ha="left", va="top", fontsize=7.5, color=style.INK)

axb.set_xlim(-0.01, 1.01)
axb.set_xticks(np.linspace(0, 1, 6))
axb.set_xlabel("position in the chain of thought (fraction)")
fig.suptitle("Statements of intent come after the lock; honesty claims come with it",
             fontsize=10, y=0.995)
style.save(fig, "F5_timing")

if __name__ == "__main__":
    print(f"deniers n={n}: median lock {np.median(lock_d):.3f}, first claim {np.median(first):.3f}, "
          f"last claim {np.median(last):.3f}, first claim after lock {after.mean():.3f}")
    print(f"intent n={m}: median intent {np.median(ipos):.3f}, median lock (same units) {np.median(lock_i):.3f}, "
          f"intent after lock {np.mean(ipos > lock_i):.3f}")
    print("lock medians by label (all 250): " + ", ".join(
        f"{L} {np.median([S[r['source_id']]['commitment_frac'] for r in pdr if r['grid_idx'] == 5 and r['denial_label'] == L]):.3f}"
        for L in ("Admits", "Denies", "Mentions")))
