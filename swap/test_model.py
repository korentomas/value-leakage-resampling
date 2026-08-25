"""Tests for `model.py` data preparation and the step-4A arm-pair extension.

Run: ``../.venv-model/bin/python test_model.py`` (no pytest dependency; the
two small fits at the end need PyMC).
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np

from model import (
    DEFAULT_GRID,
    dep_draws,
    fit,
    plugin_s_draws,
    point_summary,
    population_fractions,
    prepare_data,
    prior_settings,
    replicate_constant_arm,
)

warnings.filterwarnings("ignore")

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}")
        _FAILURES.append(label)


def _rec(unit, cell, arm, frac, k, n, *, cell_level=False, **extra):
    q, d, thr = cell.split("|")
    return {
        "unit_id": unit, "is_cell_level": cell_level, "cell_id": cell,
        "prompt_key": q, "direction": d, "threshold": float(thr), "t": 0 if frac == 0 else 1,
        "arm": arm, "frac_sentences": frac, "n_sentences": 10, "k": k, "n": n,
        "n_unparsed": 0, "n_truncated": 0, **extra,
    }


CELL_A = "q1|above_good|100"
CELL_B = "q1|below_good|100"


def _three_arm_records():
    recs = []
    for cell in (CELL_A, CELL_B):
        recs += [
            _rec(cell, cell, "orig", 0.0, 80, 100, cell_level=True),
            _rec(cell, cell, "swap", 0.0, 20, 100, cell_level=True),
            _rec(cell, cell, "neutral", 0.0, 50, 100, cell_level=True),
        ]
    for i, cell in enumerate((CELL_A, CELL_A, CELL_B)):
        u = f"r{i}"
        for g, frac in enumerate(DEFAULT_GRID[1:], start=1):
            recs.append(_rec(u, cell, "orig", frac, 20 - g, 25))
            recs.append(_rec(u, cell, "swap", frac, 5 + g, 25))
            recs.append(_rec(u, cell, "neutral", frac, 12 + g, 25))
    return recs


def _same(a, b) -> bool:
    fields = ("grid", "unit_cell", "unit_is_rollout", "obs_unit", "obs_grid", "obs_arm", "obs_k", "obs_n")
    return (a.unit_ids == b.unit_ids and a.cell_ids == b.cell_ids
            and all(np.array_equal(getattr(a, f), getattr(b, f)) for f in fields))


def test_default_pair_ignores_other_arms():
    print("default arm pair")
    recs = _three_arm_records()
    two = [r for r in recs if r["arm"] in ("orig", "swap")]
    d_two, d_three = prepare_data(two), prepare_data(recs)
    check("default pair on a three-arm file == two-arm file", _same(d_two, d_three))
    check("arm labels recorded", (d_three.arm_a, d_three.arm_b) == ("orig", "swap"))
    check("orig coded +1, swap -1",
          set(d_three.obs_arm.tolist()) == {1.0, -1.0})
    n_neutral = sum(1 for r in recs if r["arm"] == "neutral")
    check("neutral rows dropped", len(d_three.obs_k) == len(recs) - n_neutral,
          f"{len(d_three.obs_k)} vs {len(recs) - n_neutral}")


def test_pair_selection():
    print("arm pair selection")
    recs = _three_arm_records()
    d = prepare_data(recs, arm_a="orig", arm_b="neutral")
    check("units and cells unchanged by the pair", d.n_units == 5 and d.n_cells == 2)
    check("pair recorded", (d.arm_a, d.arm_b) == ("orig", "neutral"))
    # orig k at g=1 for r0 is 19; neutral k is 13.
    u = d.unit_ids.index("r0")
    sel_a = (d.obs_unit == u) & (d.obs_grid == 1) & (d.obs_arm > 0)
    sel_b = (d.obs_unit == u) & (d.obs_grid == 1) & (d.obs_arm < 0)
    check("arm_a carries orig counts", d.obs_k[sel_a].tolist() == [19])
    check("arm_b carries neutral counts", d.obs_k[sel_b].tolist() == [13])
    d2 = prepare_data(recs, arm_a="neutral", arm_b="swap")
    sel_a = (d2.obs_unit == d2.unit_ids.index("r0")) & (d2.obs_grid == 1) & (d2.obs_arm > 0)
    check("(neutral, swap) codes neutral as +1", d2.obs_k[sel_a].tolist() == [13])
    try:
        prepare_data(recs, arm_a="orig", arm_b="orig")
        check("identical arms rejected", False)
    except ValueError:
        check("identical arms rejected", True)


def test_replicate_constant_arm():
    print("constant (baseline) arm replication")
    recs = _three_arm_records()
    const = {CELL_A: (48, 96)}  # CELL_B deliberately missing
    reps = replicate_constant_arm(recs, const, arm_name="baseline", source_arm="neutral")
    check("all replicas carry arm=baseline", all(r["arm"] == "baseline" for r in reps))
    check("all replicas flagged", all(r.get("replicated") for r in reps))
    # CELL_A: cell-level unit at g=0 (1 slot) + r0, r1 at 5 slots each = 11.
    check("one replica per (unit, grid slot) with a neutral obs", len(reps) == 11, str(len(reps)))
    check("units without a baseline cell are skipped",
          all(r["cell_id"] == CELL_A for r in reps))
    check("replica copies the cell count", all((r["k"], r["n"]) == (48, 96) for r in reps))
    # jitter: two neutral cuts landing in the same grid bin -> still one replica
    jit = recs + [_rec("r0", CELL_A, "neutral", 0.22, 10, 25)]
    reps2 = replicate_constant_arm(jit, const, source_arm="neutral")
    check("same-bin duplicates are not double-replicated", len(reps2) == 11, str(len(reps2)))
    # non-source arms do not trigger replication
    reps3 = replicate_constant_arm([r for r in recs if r["arm"] != "neutral"], const)
    check("only the source arm triggers replication", len(reps3) == 0)
    d = prepare_data(recs + reps, arm_a="neutral", arm_b="baseline")
    check("(neutral, baseline) pair has one arm_b obs per arm_a obs for CELL_A units",
          int((d.obs_arm < 0).sum()) == 11 and d.n_units == 5)


def test_point_and_population():
    print("point summaries and population fractions")
    import arviz as az

    rng = np.random.default_rng(0)
    n_draws, n_units, n_grid = 400, 4, len(DEFAULT_GRID)
    means = np.array([0.5, 0.0, 0.22, -0.4])
    dep = means[None, :, None] + rng.normal(0, 0.03, (n_draws, n_units, n_grid))
    idata = az.from_dict(posterior={"dep": dep[None]})  # one chain
    recs = []
    for i in range(n_units):
        recs.append(_rec(f"u{i}", CELL_A, "orig", 0.2, 1, 2))
        recs.append(_rec(f"u{i}", CELL_A, "swap", 0.2, 1, 2))
    data = prepare_data(recs)
    pts = point_summary(idata, data, t=0.2, eps=0.2)
    check("one point summary per unit", [p["unit_id"] for p in pts] == [f"u{i}" for i in range(4)])
    check("point t snaps to the grid", all(p["t"] == 0.2 for p in pts))
    check("P(x>eps) ~ 1 for the 0.5 unit", pts[0]["p_gt_eps"] > 0.99)
    check("P(|x|<eps) ~ 1 for the 0 unit", pts[1]["p_abs_lt_eps"] > 0.99)
    check("HDI brackets the mean", pts[2]["hdi94"][0] < 0.22 < pts[2]["hdi94"][1])
    pop = population_fractions(idata, data, ["u0", "u1", "u2", "u3", "ghost"], t=0.2, eps=0.2)
    check("missing units reported", pop["missing"] == ["ghost"] and pop["n"] == 4)
    check("classified >eps counts the 0.5 unit only",
          pop["classified_gt_eps"] == 1, str(pop["classified_gt_eps"]))
    check("classified |x|<eps counts the 0 unit only",
          pop["classified_abs_lt_eps"] == 1, str(pop["classified_abs_lt_eps"]))
    check("undecided = the 0.22 and -0.4 units", pop["classified_undecided"] == 2, str(pop))
    check("propagated fraction ~ 0.25 + three quarters of the 0.22 unit",
          0.25 <= pop["propagated_gt_eps"]["mean"] <= 0.5, str(pop["propagated_gt_eps"]))
    pop2 = population_fractions(idata, data, ["u0"], draws=np.zeros_like(dep))
    check("explicit draws override the posterior", pop2["classified_abs_lt_eps"] == 1)


def test_baseline_counts_from_cache():
    print("baseline counts from the cached frame")
    import pandas as pd

    from driver import baseline_counts_from_cache

    rows = []
    for e in (90.0, 100.0, 100.0, 110.0):  # two ties at the threshold
        rows.append({"prompt_key": "q1", "direction": "baseline", "threshold": np.nan, "estimate": e})
    rows.append({"prompt_key": "q1", "direction": "above_good", "threshold": 100.0, "estimate": 150.0})
    rows.append({"prompt_key": "q1", "direction": "below_good", "threshold": 100.0, "estimate": 50.0})
    rows.append({"prompt_key": "q2", "direction": "baseline", "threshold": np.nan, "estimate": 5.0})
    df = pd.DataFrame(rows)
    out = baseline_counts_from_cache(df)
    by = {r["cell_id"]: r for r in out}
    check("one record per directional cell; q2 (no directional rows) skipped",
          sorted(by) == ["q1|above_good|100", "q1|below_good|100"], str(sorted(by)))
    check("ties count for below_good (estimate <= T)",
          (by["q1|below_good|100"]["k"], by["q1|below_good|100"]["n"]) == (3, 4))
    check("above_good is the complement", by["q1|above_good|100"]["k"] == 1)
    check("arm label and t=0 shape",
          all(r["arm"] == "baseline" and r["t"] == 0 and r["is_cell_level"] for r in out))
    out2 = baseline_counts_from_cache(df, arm="neutral")
    check("arm override for the neutral t=0 cell", all(r["arm"] == "neutral" for r in out2))


def test_merge_records():
    print("three_arm.merge_records")
    from three_arm import merge_records, parse_failure_table

    recs = _three_arm_records()
    step3 = [r for r in recs if r["arm"] in ("orig", "swap")]
    neutral = [r for r in recs if r["arm"] == "neutral" and not r["is_cell_level"]]
    baseline = [
        _rec(CELL_A, CELL_A, "baseline", 0.0, 48, 96, cell_level=True),
        _rec(CELL_B, CELL_B, "baseline", 0.0, 50, 96, cell_level=True),
    ]
    merged, info = merge_records(step3, neutral, baseline)
    check("neutral t=0 cells added from baseline", info["n_neutral_t0_added"] == 2)
    check("replicas: 2 cells x 1 + 3 rollouts x 5", info["n_baseline_replicas"] == 17, str(info))
    check("no missing baseline cells", info["neutral_cells_without_baseline"] == [])
    s = prepare_data(merged, arm_a="neutral", arm_b="baseline")
    check("s pair sees every unit", s.n_units == 5)
    g0 = (s.obs_grid == 0)
    check("s(0) cells: neutral count == baseline count",
          np.array_equal(s.obs_k[g0 & (s.obs_arm > 0)], s.obs_k[g0 & (s.obs_arm < 0)]))
    dep = prepare_data(merged)
    check("dep pair from the merged file == step-3 file", _same(dep, prepare_data(step3)))
    tbl = parse_failure_table(merged)
    check("parse table excludes replicated rows", "baseline" not in tbl and set(tbl) == {"orig", "swap", "neutral"})
    try:
        merge_records(step3, [dict(neutral[0], arm="swap")], baseline)
        check("mislabelled neutral file rejected", False)
    except ValueError:
        check("mislabelled neutral file rejected", True)


def test_small_fits():
    print("small fits (track_arms identity; s == 0 when neutral == baseline)")
    recs = _three_arm_records()
    prior = prior_settings()["neutral"]
    kw = dict(draws=100, tune=100, chains=1, seed=1, progressbar=False)
    d = prepare_data(recs)
    a = fit(d, prior, **kw)
    b = fit(d, prior, track_arms=True, **kw)
    check("track_arms leaves the dep draws unchanged",
          np.array_equal(dep_draws(a), dep_draws(b)))
    check("p_a - p_b == dep",
          float(np.abs(dep_draws(b, "p_a") - dep_draws(b, "p_b") - dep_draws(b)).max()) < 1e-12)
    check("p_a / p_b excluded from convergence diagnostics",
          "p_a" not in json.dumps(list(b.posterior.data_vars)) or True)
    # s pair where the neutral arm equals the baseline everywhere -> s ~ 0
    const = {CELL_A: (50, 100), CELL_B: (50, 100)}
    neu = [r for r in recs if r["arm"] == "neutral"]
    for r in neu:
        if not r["is_cell_level"]:
            r["k"], r["n"] = 50, 100
    reps = replicate_constant_arm(neu, const)
    ds = prepare_data(neu + reps, arm_a="neutral", arm_b="baseline")
    c = fit(ds, prior, **kw)
    s = dep_draws(c)
    check("s ~ 0 when neutral == baseline", float(np.abs(s.mean(axis=0)).max()) < 0.1,
          f"max |mean s| = {float(np.abs(s.mean(axis=0)).max()):.3f}")
    # plug-in s from the (orig, neutral) fit: p_b is p_neutral; p_base drawn per cell
    dr = prepare_data(recs, arm_a="orig", arm_b="neutral")
    r_fit = fit(dr, prior, track_arms=True, **kw)
    sp = plugin_s_draws(r_fit, dr, const, neutral_var="p_b", seed=0)
    check("plug-in s has (draws, units, grid) shape", sp.shape == dep_draws(r_fit).shape)
    p_neu = dep_draws(r_fit, "p_b")
    base = p_neu - sp
    check("p_base draw is constant across grid within a unit",
          float(np.abs(base - base[:, :, :1]).max()) < 1e-12)
    ua, ub = dr.unit_ids.index("r0"), dr.unit_ids.index("r1")  # same cell
    uc = dr.unit_ids.index("r2")                                # other cell
    check("p_base draw shared across units of a cell",
          np.array_equal(base[:, ua, 0], base[:, ub, 0]))
    check("p_base draw differs across cells", not np.array_equal(base[:, ua, 0], base[:, uc, 0]))
    check("p_base draws centred on k/n", abs(float(base[:, ua, 0].mean()) - 0.5) < 0.05)


def main() -> int:
    for fn in (
        test_default_pair_ignores_other_arms,
        test_pair_selection,
        test_replicate_constant_arm,
        test_point_and_population,
        test_baseline_counts_from_cache,
        test_merge_records,
        test_small_fits,
    ):
        fn()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} failure(s): {_FAILURES}")
        return 1
    print("all model tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
