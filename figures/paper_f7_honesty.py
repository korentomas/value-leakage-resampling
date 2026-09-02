"""Paper F7 - timing of the first honesty claim relative to the lock.

The same two panels as F6, on a different quantity, so the two can be compared
directly. F6 plots stated intent; this plots honesty claims such as "I must be
accurate", located in the 77 denying conversations where one could be found.

The point of the pair: intent statements come after the lock in 81% of cases,
honesty claims in 45%. The first is retrospective, the second arrives with the
decision.

Run:  ../.venv-model/bin/python paper_f7_honesty.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps

ps.use()

S = {u["unit_id"]: u for u in json.load(open("../artifacts/step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
den = [json.loads(l) for l in open("../artifacts/denial_positions.jsonl")]
den = [r for r in den if r["rollout_id"] in S]
lock = np.array([S[r["rollout_id"]]["commitment_frac"] for r in den])
claim = np.array([r["first_frac"] for r in den])
diff = claim - lock
after = float(np.mean(diff > 0))

fig, axes = plt.subplots(1, 2, figsize=ps.WIDE)
bins = np.linspace(0, 1, 21)

ax = axes[0]
ax.hist(lock, bins=bins, color=ps.ORIG, alpha=0.65, label="answer locked", edgecolor="white", lw=0.4)
ax.hist(claim, bins=bins, color=ps.SWAP, alpha=0.65, label="first honesty claim", edgecolor="white", lw=0.4)
ax.set_xlabel("position in the CoT")
ax.set_ylabel("conversations")
ax.set_title("(a) positions")

ax = axes[1]
ax.hist(diff, bins=np.linspace(-1, 1, 25), color=ps.NEUTRAL, edgecolor="white", lw=0.4)
ax.axvline(0, color=ps.REF, ls="--", lw=1.2)
ax.set_xlabel("first claim $-$ lock")
ax.set_ylabel("conversations")
ax.set_title("(b) per-conversation difference")

fig.tight_layout()
ps.below_legend(fig, axes[0], ncol=2, y=0.02)
ps.save(fig, "F7_honesty")

if __name__ == "__main__":
    print(f"n={len(den)} median lock {np.median(lock):.3f} median claim {np.median(claim):.3f} after {after:.3f}")
