"""Export TRUE posterior quantities for the figures, from the saved idata.

Everything the notebook plots comes from here — no Beta approximations,
no placeholders. Classification uncertainty is propagated: a unit's
steered-status is evaluated per posterior draw (any grid>0 with dep>eps),
so rates among deniers/admitters/questions are full posteriors.

Run (fitting env, from figures/):
    ../.venv-model/bin/python export_truth.py
Writes figure_truth.npz + prints headline checks.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "../swap")
import arviz as az  # noqa: E402
from model import prepare_data, load_counts  # noqa: E402

EPS = 0.2
ART = Path("../artifacts")

data = prepare_data(load_counts(ART / "step3_counts.jsonl"))
rollout_ix = np.where(data.unit_is_rollout)[0]
unit_ids = [data.unit_ids[i] for i in rollout_ix]
cell_ids = [data.cell_ids[int(data.unit_cell[i])] for i in rollout_ix]
qs = [c.split("|")[0].replace("v1_", "").replace("_accurate", "") for c in cell_ids]

labels = json.load(open(ART / "denial_labels.json"))
denial = np.array([labels.get(u, "NA") for u in unit_ids])

out = {}
for prior in ("skeptical", "neutral", "informed"):
    idata = az.from_netcdf(ART / f"step3_idata_{prior}.nc")
    dep = idata.posterior["dep"].values  # (chains, draws, units, grid)
    dep = dep.reshape(-1, dep.shape[-2], dep.shape[-1])[:, rollout_ix, :]
    # per-draw steered status: dep > eps at any grid point past t0
    steered = (dep[:, :, 1:] > EPS).any(axis=2)  # (draws, units)

    out[f"{prior}_rate_draws"] = steered.mean(axis=1)
    if prior == "neutral":
        # population dep(t) curve: mean over units per draw
        curve = dep.mean(axis=1)  # (draws, grid)
        out["grid"] = np.array(data.grid)
        out["curve_mean"] = curve.mean(axis=0)
        out["curve_lo94"] = np.percentile(curve, 3, axis=0)
        out["curve_hi94"] = np.percentile(curve, 97, axis=0)
        # denial-conditional rates, fully propagated
        for label, mask in [("den", denial == "NOT_INFLUENCED"),
                            ("adm", denial == "INFLUENCED")]:
            out[f"{label}_rate_draws"] = steered[:, mask].mean(axis=1)
        out["diff_draws"] = out["den_rate_draws"] - out["adm_rate_draws"]
        # per-question rates
        for q in sorted(set(qs)):
            mask = np.array([x == q for x in qs])
            out[f"q_{q}_draws"] = steered[:, mask].mean(axis=1)
            out[f"q_{q}_n"] = int(mask.sum())

np.savez(Path("figure_truth.npz"), **out)

n = out["neutral_rate_draws"]
d = out["diff_draws"]
print(f"neutral population rate: mean {n.mean():.3f}, 94% HDI "
      f"[{np.percentile(n,3):.3f}, {np.percentile(n,97):.3f}]")
print(f"P(steered|denies) {out['den_rate_draws'].mean():.3f} vs "
      f"P(steered|admits) {out['adm_rate_draws'].mean():.3f}")
print(f"difference 94% HDI [{np.percentile(d,3):.3f}, {np.percentile(d,97):.3f}], "
      f"P(>0) = {(d>0).mean():.2f}")
print("wrote figure_truth.npz")
