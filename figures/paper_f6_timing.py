"""Paper F6 - timing of the first stated intention relative to the lock.

Replaces the blog jitter strips. Panel (a) overlays the two position
distributions; panel (b) is the per-conversation difference, which is the
paired quantity the claim is actually about, with zero marked.

Run:  ../.venv-model/bin/python paper_f6_timing.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps

ps.use()

S = {u["unit_id"]: u for u in json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
pdr = [json.loads(l) for l in open("../artifacts/prefix_direction.jsonl")]
intent = [r for r in pdr if r["grid_idx"] == 5 and r["has_intent"] and r["intent_pos_full"] is not None]
lock = np.array([S[r["source_id"]]["commitment_frac"] for r in intent])
ipos = np.array([r["intent_pos_full"] for r in intent])
diff = ipos - lock
after = float(np.mean(diff > 0))

fig, axes = plt.subplots(1, 2, figsize=ps.WIDE)
bins = np.linspace(0, 1, 21)

ax = axes[0]
ps.grid(ax)
ax.hist(lock, bins=bins, color=ps.ORIG, alpha=0.65, label="answer locked", edgecolor="white", lw=0.4)
ax.hist(ipos, bins=bins, color=ps.SWAP, alpha=0.65, label="first stated intent", edgecolor="white", lw=0.4)
ax.set_xlabel("position in chain of thought (fraction)")
ax.set_ylabel("conversations")
ax.set_title("(a) positions")
ax.legend()

ax = axes[1]
ps.grid(ax)
ax.hist(diff, bins=np.linspace(-1, 1, 25), color=ps.NEUTRAL, edgecolor="white", lw=0.4)
ax.axvline(0, color=ps.REF, ls="--", lw=1.2, label=f"intent after lock: {after:.0%}")
ax.set_xlabel("first intent $-$ lock (fraction)")
ax.set_ylabel("conversations")
ax.set_title("(b) per-conversation difference")
ax.legend()

fig.tight_layout()
ps.save(fig, "F6_timing")

if __name__ == "__main__":
    print(f"n={len(intent)} median lock {np.median(lock):.3f} median intent {np.median(ipos):.3f} after {after:.3f}")
