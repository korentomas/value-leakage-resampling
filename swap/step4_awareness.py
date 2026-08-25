"""4A follow-up: is the denier/admitter bet-shaped gap explained by verbalised
eval-awareness (paper Fig 30)? Stratify the neutral-arm bet-shaped posterior by
the paper's awareness labels (artifacts/qwen3.5-35_master.parquet)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, arviz as az
from model import DEFAULT_GRID, load_counts, prepare_data
from three_arm import PAIRS, merge_records, shared_baseline_s_draws
A = Path("../artifacts")
def eti(x): return (float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
step3 = load_counts(A/"step3_counts.jsonl"); neutral = load_counts(A/"step4_neutral_counts.jsonl"); base = load_counts(A/"step4_baseline_counts.jsonl")
merged, _ = merge_records(step3, neutral, base); cbc = {r["cell_id"]: (int(r["k"]), int(r["n"])) for r in base}
data = prepare_data(merged, grid=DEFAULT_GRID, arm_a=PAIRS["r"][0], arm_b=PAIRS["r"][1])
idata = az.from_netcdf(A/"step4_fits/neutral/step4_idata_r_neutral.nc")
s = shared_baseline_s_draws(idata, data, cbc, pair="r")[:, :, 1]
units = list(data.unit_ids)
st3 = {u["unit_id"]: u for u in json.load(open(A/"step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
df = pd.read_json(A/"qwen3.5-35_labels.json").set_index("rollout_id")  # exported from the master parquet (no pyarrow in .venv-model)
idx = [i for i, u in enumerate(units) if u in st3 and u in df.index]
late = np.array([st3[units[i]]["dependent_after_t0"] for i in idx])
S = s[:, idx]; shaped = (S > 0.2) | late[None, :]
lab = np.array([df.loc[units[i], "denial_bucket"] for i in idx])
aware = np.array([bool(df.loc[units[i], "eval_aware"]) for i in idx])
weak = np.array([bool(df.loc[units[i], "weak_signal"]) for i in idx])
score = np.array([df.loc[units[i], "awareness_score"] for i in idx], dtype=float)
print("n", len(idx), "labels", dict(zip(*np.unique(lab, return_counts=True))), "eval_aware", aware.sum(), "weak", weak.sum())
out = {}
def row(name, m):
    if m.sum() == 0: return
    b = eti(shaped[:, m].mean(1)); sm = eti(S[:, m].mean(1)); out[name] = {"n": int(m.sum()), "bet_shaped": b, "s_mean": sm}
    print(f"  {name:38s} n={m.sum():3d}  bet-shaped {b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]  s(.2) {sm[0]:+.3f} [{sm[1]:+.3f},{sm[2]:+.3f}]")
print("by awareness (score>=4):")
for nm, m in (("aware", aware), ("not aware", ~aware), ("weak signal (>=2)", weak), ("no signal", ~weak)): row(nm, m)
print("awareness by label: Denies aware %.3f (n=%d), Admits aware %.3f (n=%d)" % (aware[lab=="Denies"].mean(), (lab=="Denies").sum(), aware[lab=="Admits"].mean(), (lab=="Admits").sum()))
print("label x awareness strata:")
for L in ("Denies", "Admits"):
    for nm, m in (("aware", aware), ("not aware", ~aware)): row(f"{L} | {nm}", (lab == L) & m)
def contrast(mask, name):
    d = (lab == "Denies") & mask; a = (lab == "Admits") & mask
    if d.sum() < 3 or a.sum() < 3: return
    c = eti(shaped[:, d].mean(1) - shaped[:, a].mean(1)); cs = eti(S[:, d].mean(1) - S[:, a].mean(1))
    out[f"contrast {name}"] = {"n_den": int(d.sum()), "n_adm": int(a.sum()), "bet_shaped_diff": c, "s_diff": cs}
    print(f"  Denies-Admits within {name:12s} (n={d.sum()}/{a.sum()}): bet-shaped {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]  s {cs[0]:+.3f} [{cs[1]:+.3f},{cs[2]:+.3f}]")
print("contrast within strata:")
contrast(np.ones_like(aware), "all"); contrast(aware, "aware"); contrast(~aware, "not aware"); contrast(weak, "weak>=2"); contrast(~weak, "no signal")
# awareness-adjusted contrast: weighted average of within-stratum contrasts (weights = pooled stratum share)
w = np.array([aware.mean(), 1 - aware.mean()])
dA = (shaped[:, (lab=="Denies")&aware].mean(1) - shaped[:, (lab=="Admits")&aware].mean(1))
dN = (shaped[:, (lab=="Denies")&~aware].mean(1) - shaped[:, (lab=="Admits")&~aware].mean(1))
adj = eti(w[0]*dA + w[1]*dN); out["contrast awareness-adjusted"] = adj
print(f"  awareness-adjusted Denies-Admits (bet-shaped): {adj[0]:+.3f} [{adj[1]:+.3f},{adj[2]:+.3f}]")
json.dump(out, open(A/"step4_awareness.json", "w"), indent=1)
