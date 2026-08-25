#!/usr/bin/env python3
"""STEP4.md 4C critique addenda: data report only, no writeup edits.

Run: ../external/value_leakage/.venv/bin/python critique_addenda.py
(from inside swap/), or any python3 with numpy on the path (part 5 only
needs numpy; parts 1-4,6,7 use only stdlib json/statistics + numpy for HDI).
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures"

out = {}
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def quartiles(xs):
    if not xs:
        return (float("nan"),) * 3
    xs = sorted(xs)
    return (
        np.percentile(xs, 25),
        np.percentile(xs, 50),
        np.percentile(xs, 75),
    )


def hdi94(draws):
    draws = np.sort(np.asarray(draws))
    n = len(draws)
    k = int(np.floor(0.94 * n))
    if k >= n:
        return draws[0], draws[-1]
    widths = draws[k:] - draws[: n - k]
    j = int(np.argmin(widths))
    return draws[j], draws[j + k]


# ---------------------------------------------------------------------
# 1. Honesty-claim timing, FIRST vs LAST claim
# ---------------------------------------------------------------------
p("## 1. Honesty-claim timing: FIRST vs LAST claim vs commitment\n")

denial_positions = {}
with open(ART / "denial_positions.jsonl") as f:
    for line in f:
        r = json.loads(line)
        denial_positions[r["rollout_id"]] = r

summaries = json.load(open(ART / "step3_summaries.json"))["neutral"]["summaries"]
summ_by_unit = {s["unit_id"]: s for s in summaries if s.get("is_rollout")}

joined = []
for rid, dp in denial_positions.items():
    s = summ_by_unit.get(rid)
    if s is None:
        continue
    joined.append(
        {
            "rollout_id": rid,
            "first_frac": dp["first_frac"],
            "last_frac": dp["last_frac"],
            "commitment_frac": s["commitment_frac"],
            "dependent_after_t0": s["dependent_after_t0"],
        }
    )

n_joined = len(joined)
first_after = [j for j in joined if j["first_frac"] > j["commitment_frac"]]
last_after = [j for j in joined if j["last_frac"] > j["commitment_frac"]]

p(f"n joined (denial_positions x step3_summaries[neutral].summaries, unit_id/rollout_id): {n_joined}")
p(f"first_frac > commitment_frac: {len(first_after)}/{n_joined} = {len(first_after)/n_joined:.3f}" if n_joined else "n_joined=0")
p(f"last_frac  > commitment_frac: {len(last_after)}/{n_joined} = {len(last_after)/n_joined:.3f}" if n_joined else "")
p("")
p(f"median first_frac       = {median([j['first_frac'] for j in joined]):.4f}")
p(f"median last_frac        = {median([j['last_frac'] for j in joined]):.4f}")
p(f"median commitment_frac  = {median([j['commitment_frac'] for j in joined]):.4f}")
p("")

expected = {"n_joined": 77, "first_after": 35, "last_after": 71}
p(f"Expected (prior hand computation): n=77, first-after=35/77, last-after=71/77")
disc1 = []
if n_joined != expected["n_joined"]:
    disc1.append(f"n_joined {n_joined} != expected {expected['n_joined']}")
if len(first_after) != expected["first_after"]:
    disc1.append(f"first_after {len(first_after)} != expected {expected['first_after']}")
if len(last_after) != expected["last_after"]:
    disc1.append(f"last_after {len(last_after)} != expected {expected['last_after']}")
if disc1:
    p("DISCREPANCY: " + "; ".join(disc1))
else:
    p("Matches expected numbers exactly.")
p("")

p("### split by dependent_after_t0 (markdown)\n")
p("| dependent_after_t0 | n | median first_frac | median last_frac | median commitment_frac | first>commit | last>commit |")
p("|---|---|---|---|---|---|---|")
table1_split = {}
for dep_flag in (True, False):
    sub = [j for j in joined if j["dependent_after_t0"] == dep_flag]
    if not sub:
        p(f"| {dep_flag} | 0 | - | - | - | - | - |")
        continue
    fa = sum(1 for j in sub if j["first_frac"] > j["commitment_frac"])
    la = sum(1 for j in sub if j["last_frac"] > j["commitment_frac"])
    p(
        f"| {dep_flag} | {len(sub)} | {median([j['first_frac'] for j in sub]):.3f} "
        f"| {median([j['last_frac'] for j in sub]):.3f} | {median([j['commitment_frac'] for j in sub]):.3f} "
        f"| {fa}/{len(sub)} | {la}/{len(sub)} |"
    )
    table1_split[str(dep_flag)] = {
        "n": len(sub),
        "median_first_frac": median([j["first_frac"] for j in sub]),
        "median_last_frac": median([j["last_frac"] for j in sub]),
        "median_commitment_frac": median([j["commitment_frac"] for j in sub]),
        "first_after_commit": fa,
        "last_after_commit": la,
    }
p("")

out["honesty_timing"] = {
    "n_joined": n_joined,
    "first_after_commit": len(first_after),
    "last_after_commit": len(last_after),
    "median_first_frac": median([j["first_frac"] for j in joined]),
    "median_last_frac": median([j["last_frac"] for j in joined]),
    "median_commitment_frac": median([j["commitment_frac"] for j in joined]),
    "expected": expected,
    "discrepancies": disc1,
    "split_by_dependent_after_t0": table1_split,
}

# ---------------------------------------------------------------------
# load step3_counts.jsonl once, used by parts 2, 3, 4
# ---------------------------------------------------------------------
counts = [json.loads(l) for l in open(ART / "step3_counts.jsonl")]
grid_counts = [r for r in counts if not r["is_cell_level"] and r["t"] != 0]
cell_t0 = [r for r in counts if r["is_cell_level"] and r["t"] == 0]


def grid_idx(frac):
    return round(frac * 5) / 5


# ---------------------------------------------------------------------
# 2. Unparsed rate by arm x grid cut
# ---------------------------------------------------------------------
p("\n## 2. Unparsed rate by arm x grid cut (step3_counts.jsonl, grid rows only, t!=0)\n")

by_arm_cut = defaultdict(lambda: {"n": 0, "n_unparsed": 0})
by_arm_total = defaultdict(lambda: {"n": 0, "n_unparsed": 0, "n_truncated": 0})
for r in grid_counts:
    gi = grid_idx(r["frac_sentences"])
    key = (r["arm"], gi)
    by_arm_cut[key]["n"] += r["n"]
    by_arm_cut[key]["n_unparsed"] += r["n_unparsed"]
    by_arm_total[r["arm"]]["n"] += r["n"]
    by_arm_total[r["arm"]]["n_unparsed"] += r["n_unparsed"]
    by_arm_total[r["arm"]]["n_truncated"] += r["n_truncated"]

arms = sorted(by_arm_total.keys())
grids = sorted({k[1] for k in by_arm_cut})

p("| arm | grid | n | n_unparsed | rate | n_cells |")
p("|---|---|---|---|---|---|")
table2 = {}
for arm in arms:
    for gi in grids:
        d = by_arm_cut.get((arm, gi))
        if not d:
            continue
        denom = d["n"] + d["n_unparsed"]
        rate = d["n_unparsed"] / denom if denom else float("nan")
        p(f"| {arm} | {gi} | {d['n']} | {d['n_unparsed']} | {rate:.4f} |")
        table2.setdefault(arm, {})[str(gi)] = {
            "n": d["n"],
            "n_unparsed": d["n_unparsed"],
            "rate": rate,
        }

p("\n### totals per arm\n")
p("| arm | n | n_unparsed | n_truncated | rate |")
p("|---|---|---|---|---|")
totals2 = {}
for arm in arms:
    d = by_arm_total[arm]
    denom = d["n"] + d["n_unparsed"]
    rate = d["n_unparsed"] / denom if denom else float("nan")
    p(f"| {arm} | {d['n']} | {d['n_unparsed']} | {d['n_truncated']} | {rate:.4f} |")
    totals2[arm] = {**d, "rate": rate}

expected2 = {"swap": 0.158, "orig": 0.144}
disc2 = []
for arm, exp in expected2.items():
    got = totals2.get(arm, {}).get("rate", float("nan"))
    if abs(got - exp) > 0.001:
        disc2.append(f"{arm}: got {got:.4f} vs expected {exp}")
p(f"\nExpected: swap 0.158, orig 0.144")
if disc2:
    p("DISCREPANCY: " + "; ".join(disc2))
else:
    p("Matches expected numbers.")

out["unparsed_rate"] = {"by_arm_grid": table2, "totals": totals2, "expected": expected2, "discrepancies": disc2}

# ---------------------------------------------------------------------
# 3. Raw favoured-side rate by direction x arm x grid cut; raw dep
# ---------------------------------------------------------------------
p("\n## 3. Raw (model-free) favoured-side rate by direction x arm x grid cut\n")

by_dir_arm_cut = defaultdict(lambda: {"k": 0, "n": 0})
for r in grid_counts:
    gi = grid_idx(r["frac_sentences"])
    key = (r["direction"], r["arm"], gi)
    by_dir_arm_cut[key]["k"] += r["k"]
    by_dir_arm_cut[key]["n"] += r["n"]

directions = sorted({k[0] for k in by_dir_arm_cut})
p("| direction | arm | grid | k | n | rate |")
p("|---|---|---|---|---|---|")
table3_raw = {}
for d_ in directions:
    for arm in arms:
        for gi in grids:
            dd = by_dir_arm_cut.get((d_, arm, gi))
            if not dd or dd["n"] == 0:
                continue
            rate = dd["k"] / dd["n"]
            p(f"| {d_} | {arm} | {gi} | {dd['k']} | {dd['n']} | {rate:.4f} |")
            table3_raw.setdefault(d_, {}).setdefault(arm, {})[str(gi)] = {
                "k": dd["k"],
                "n": dd["n"],
                "rate": rate,
            }

p("\n### raw dep(t) = rate(orig) - rate(swap), by direction and pooled\n")
p("| direction | grid | rate_orig | rate_swap | dep |")
p("|---|---|---|---|---|")
table3_dep = {}
pooled_cut = defaultdict(lambda: {"k_o": 0, "n_o": 0, "k_s": 0, "n_s": 0})
for d_ in directions:
    for gi in grids:
        o = by_dir_arm_cut.get((d_, "orig", gi))
        s = by_dir_arm_cut.get((d_, "swap", gi))
        if not o or not s or o["n"] == 0 or s["n"] == 0:
            continue
        ro, rs = o["k"] / o["n"], s["k"] / s["n"]
        p(f"| {d_} | {gi} | {ro:.4f} | {rs:.4f} | {ro - rs:.4f} |")
        table3_dep.setdefault(d_, {})[str(gi)] = {"rate_orig": ro, "rate_swap": rs, "dep": ro - rs}
        pooled_cut[gi]["k_o"] += o["k"]
        pooled_cut[gi]["n_o"] += o["n"]
        pooled_cut[gi]["k_s"] += s["k"]
        pooled_cut[gi]["n_s"] += s["n"]

p("\n### pooled (both directions combined)\n")
p("| grid | rate_orig | rate_swap | dep | n_orig | n_swap |")
p("|---|---|---|---|---|---|")
table3_pooled = {}
for gi in grids:
    c = pooled_cut[gi]
    if c["n_o"] == 0 or c["n_s"] == 0:
        continue
    ro, rs = c["k_o"] / c["n_o"], c["k_s"] / c["n_s"]
    p(f"| {gi} | {ro:.4f} | {rs:.4f} | {ro - rs:.4f} | {c['n_o']} | {c['n_s']} |")
    table3_pooled[str(gi)] = {"rate_orig": ro, "rate_swap": rs, "dep": ro - rs, "n_orig": c["n_o"], "n_swap": c["n_s"]}

out["raw_favoured_rate"] = {"by_direction_arm_cut": table3_raw, "raw_dep_by_direction": table3_dep, "raw_dep_pooled": table3_pooled}

# ---------------------------------------------------------------------
# 4. Raw t=0 bias from cell-level rows
# ---------------------------------------------------------------------
p("\n## 4. Raw t=0 bias from cell-level rows (is_cell_level True, t==0)\n")

t0_by_dir_arm = defaultdict(lambda: {"k": 0, "n": 0})
for r in cell_t0:
    t0_by_dir_arm[(r["direction"], r["arm"])]["k"] += r["k"]
    t0_by_dir_arm[(r["direction"], r["arm"])]["n"] += r["n"]

p("| direction | arm | k | n | rate |")
p("|---|---|---|---|---|")
table4_raw = {}
for d_ in directions:
    for arm in arms:
        dd = t0_by_dir_arm.get((d_, arm))
        if not dd or dd["n"] == 0:
            continue
        rate = dd["k"] / dd["n"]
        p(f"| {d_} | {arm} | {dd['k']} | {dd['n']} | {rate:.4f} |")
        table4_raw.setdefault(d_, {})[arm] = {"k": dd["k"], "n": dd["n"], "rate": rate}

p("\n### dep(0) per direction and pooled\n")
p("| direction | dep(0) = rate_orig - rate_swap |")
p("|---|---|")
table4_dep = {}
pooled_k_o = pooled_n_o = pooled_k_s = pooled_n_s = 0
for d_ in directions:
    o = t0_by_dir_arm.get((d_, "orig"))
    s = t0_by_dir_arm.get((d_, "swap"))
    if not o or not s or o["n"] == 0 or s["n"] == 0:
        continue
    ro, rs = o["k"] / o["n"], s["k"] / s["n"]
    p(f"| {d_} | {ro - rs:.4f} |")
    table4_dep[d_] = ro - rs
    pooled_k_o += o["k"]; pooled_n_o += o["n"]
    pooled_k_s += s["k"]; pooled_n_s += s["n"]

pooled_ro = pooled_k_o / pooled_n_o if pooled_n_o else float("nan")
pooled_rs = pooled_k_s / pooled_n_s if pooled_n_s else float("nan")
pooled_dep0 = pooled_ro - pooled_rs
p(f"| pooled | {pooled_dep0:.4f} |")
p(f"\nrate_orig pooled = {pooled_ro:.4f} (n={pooled_n_o}), rate_swap pooled = {pooled_rs:.4f} (n={pooled_n_s})")
p(f"Compare with fitted dep(0) = 0.617, published dep(0) = 0.62.")
table4_dep["pooled"] = pooled_dep0

out["raw_t0_bias"] = {"by_direction_arm": table4_raw, "dep0_by_direction": table4_dep, "dep0_pooled": pooled_dep0,
                       "fitted_dep0": 0.617, "published_dep0": 0.62}

# ---------------------------------------------------------------------
# 5. Population curve from figure_truth.npz
# ---------------------------------------------------------------------
p("\n## 5. Population curve (figures/figure_truth.npz)\n")

ft = np.load(FIG / "figure_truth.npz")
grid_ft = ft["grid"]
curve_mean = ft["curve_mean"]
curve_lo94 = ft["curve_lo94"]
curve_hi94 = ft["curve_hi94"]

p("| grid | curve_mean | curve_lo94 | curve_hi94 |")
p("|---|---|---|---|")
table5 = []
for g, m, lo, hi in zip(grid_ft, curve_mean, curve_lo94, curve_hi94):
    p(f"| {g:.2f} | {m:.4f} | {lo:.4f} | {hi:.4f} |")
    table5.append({"grid": float(g), "mean": float(m), "lo94": float(lo), "hi94": float(hi)})

out["population_curve"] = table5

# ---------------------------------------------------------------------
# 6. Early-locked vs late-locked units: dep_mean and p_dep_gt_eps at grid idx 1 (t=0.2)
# ---------------------------------------------------------------------
p("\n## 6. Early-locked vs late-locked: dep_mean[1] and p_dep_gt_eps[1] (t=0.2)\n")

early = [s for s in summaries if s.get("is_rollout") and s["dependent_after_t0"] is False]
late = [s for s in summaries if s.get("is_rollout") and s["dependent_after_t0"] is True]

p(f"n early-locked (dependent_after_t0=False) = {len(early)}")
p(f"n late-locked  (dependent_after_t0=True)  = {len(late)}")
p("")


def summarize_at1(units, field):
    vals = [u[field][1] for u in units]
    q1, med, q3 = quartiles(vals)
    return med, q1, q3, len(vals)


p("| group | n | field | median | Q1 | Q3 |")
p("|---|---|---|---|---|---|")
table6 = {}
for name, units in (("early-locked", early), ("late-locked", late)):
    for field in ("dep_mean", "p_dep_gt_eps"):
        med, q1, q3, n = summarize_at1(units, field)
        p(f"| {name} | {n} | {field}[1] | {med:.4f} | {q1:.4f} | {q3:.4f} |")
        table6.setdefault(name, {})[field] = {"n": n, "median": med, "q1": float(q1), "q3": float(q3)}

out["early_vs_late_locked_t02"] = table6

# ---------------------------------------------------------------------
# 7. Denier/admitter contrast, both estimands
# ---------------------------------------------------------------------
p("\n## 7. Denier/admitter contrast, both estimands\n")

den_draws = ft["den_rate_draws"]
adm_draws = ft["adm_rate_draws"]
diff_draws = ft["diff_draws"]

den_mean = float(den_draws.mean())
adm_mean = float(adm_draws.mean())
diff_mean = float(diff_draws.mean())
den_hdi = tuple(float(x) for x in hdi94(den_draws))
adm_hdi = tuple(float(x) for x in hdi94(adm_draws))
diff_hdi = tuple(float(x) for x in hdi94(diff_draws))

p("| quantity | mean | 94% HDI |")
p("|---|---|---|")
p(f"| den_rate (propagated) | {den_mean:.4f} | [{den_hdi[0]:.4f}, {den_hdi[1]:.4f}] |")
p(f"| adm_rate (propagated) | {adm_mean:.4f} | [{adm_hdi[0]:.4f}, {adm_hdi[1]:.4f}] |")
p(f"| diff (den - adm)      | {diff_mean:.4f} | [{diff_hdi[0]:.4f}, {diff_hdi[1]:.4f}] |")

p("\n### classification-based counts (restated from FINDINGS.md)\n")
p("| group | steered / total | rate |")
p("|---|---|---|")
class_counts = {
    "deniers": (34, 85),
    "admitters": (55, 151),
    "mentions": (4, 13),
}
table7_class = {}
for name, (k, n) in class_counts.items():
    rate = k / n
    p(f"| {name} | {k}/{n} | {rate:.4f} |")
    table7_class[name] = {"k": k, "n": n, "rate": rate}

out["denier_admitter_contrast"] = {
    "propagated": {
        "den_rate": {"mean": den_mean, "hdi94": den_hdi},
        "adm_rate": {"mean": adm_mean, "hdi94": adm_hdi},
        "diff": {"mean": diff_mean, "hdi94": diff_hdi},
    },
    "classification": table7_class,
}

# ---------------------------------------------------------------------
# write artifact
# ---------------------------------------------------------------------
out_path = ART / "critique_addenda.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
p(f"\nWrote {out_path}")
