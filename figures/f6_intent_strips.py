"""F6 — compact timing figure for the short LW post: the bottom panel of F5 on its own.

Two jittered strips over position in the chain of thought, for the 147 conversations
with an explicit statement of intent to fall on the good side (artifacts/prefix_direction.jsonl,
grid_idx == 5, has_intent): where the answer locked (posterior median commitment_frac, neutral
prior) and where the intent statement was written.

Run:  ../.venv-model/bin/python f6_intent_strips.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt

import style

style.use()

S = {u["unit_id"]: u for u in json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
pdr = [json.loads(l) for l in open("../artifacts/prefix_direction.jsonl")]
intent = [r for r in pdr if r["grid_idx"] == 5 and r["has_intent"] and r["intent_pos_full"] is not None]

lock = np.array([S[r["source_id"]]["commitment_frac"] for r in intent])
ipos = np.array([r["intent_pos_full"] for r in intent])
m = len(intent)
after = np.mean(ipos > lock)

fig, ax = style.plate(figsize=(8, 2.9))
rng = np.random.default_rng(0)
y_lock, y_int, jit = 1.0, 0.0, 0.17
ax.scatter(lock, y_lock + rng.uniform(-jit, jit, m), s=13, color=style.INK, alpha=0.7, lw=0, zorder=3)
ax.scatter(ipos, y_int + rng.uniform(-jit, jit, m), s=13, color=style.STEERED, alpha=0.75, lw=0, zorder=3)
for y, x, col in ((y_lock, np.median(lock), style.INK), (y_int, np.median(ipos), style.STEERED)):
    ax.plot([x, x], [y - 0.32, y + 0.32], color=col, lw=2.2, zorder=4)
    ax.text(x, y + 0.36, f"median {x:.2f}", ha="center", va="bottom", fontsize=9, color=col)

ax.set_yticks([y_lock, y_int])
ax.set_yticklabels(["answer locked", "first statement\nof intent"], fontsize=9.5)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_ylim(-0.6, 1.9)
ax.set_xlim(-0.01, 1.01)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xlabel("position in the chain of thought (fraction)")
ax.set_title("Statements of intent come after the lock",
             fontsize=11, loc="left", pad=10)
style.save(fig, "F6_intent_strips")

if __name__ == "__main__":
    print(f"n={m}, median lock {np.median(lock):.3f}, median intent {np.median(ipos):.3f}, intent after lock {after:.3f}")
