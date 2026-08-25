"""Step 4A: fit the three-arm decomposition with the step-3 model.

Inputs
------
  step-3 counts     artifacts/step3_counts.jsonl        arms orig / swap,
                                                        t>0 rollouts + t=0 cells
  neutral counts    artifacts/step4_neutral_counts.jsonl arm neutral, t>0
                                                        (driver.py aggregate on
                                                        the step-4 results)
  baseline counts   artifacts/step4_baseline_counts.jsonl arm baseline, one
                                                        record per cell
                                                        (driver.py counts-from-cache
                                                        --baseline-out)
  step-3 summaries  artifacts/step3_summaries.json      defines the early-locked
                                                        set (neutral prior,
                                                        dependent_after_t0 False)

The four records sets are merged into one list; `model.prepare_data` is then
called once per arm pair, so each pair is fitted by the *unchanged* step-3
model on the subset of arms it names:

    r = p_orig    - p_neutral      pair (orig, neutral)
    c = p_neutral - p_swap         pair (neutral, swap)
    s = p_neutral - p_base         pair (neutral, baseline)

with the baseline count replicated onto every (unit, grid slot) that has a
neutral observation (`model.replicate_constant_arm`; see the assumption
stated there). The t=0 cell of the neutral arm is the baseline record
itself — an empty prefix under the no-bet prompt *is* a baseline rollout —
so s(0) = 0 exactly and the model's w[u,0] = 0 anchor is again an identity.

Informed-prior locations per pair
---------------------------------
The step-3 informed prior centres dep(0) on the paper's published 0.62.
p_base is 0.5 by construction (thresholds are baseline medians), so
r(0) + c(0) = dep(0) with no published split; the informed location is
split evenly (0.31 each). s(0) is 0 by identity, so its informed location is
0. All three are overridable with ``--informed-dep0 r=..,c=..,s=..``.

Outputs (``--out-dir``, ``--tag`` default ``step4``)
----------------------------------------------------
  {tag}_merged_counts.jsonl             the merged record list actually fitted
  {tag}_summaries_{pair}.json           {prior: {summaries, covertness,
                                        convergence, point, early_locked}} —
                                        the step-3 layout plus the s(0.2)
                                        readings
  {tag}_summaries_{pair}_eps{e}.json    same at the other eps values
  {tag}_idata_{pair}_{prior}.nc         posterior draws
  {tag}_early_locked.json               population fractions on the
                                        early-locked set, pair x prior x eps:
                                        each pair's own estimand, plus s in
                                        three versions — replicated arm
                                        (pair s), shared-baseline draw (pair
                                        s), and plug-in from the r and c fits
                                        (p_neutral from a two-varying-arm
                                        fit minus a shared p_base draw; see
                                        model.plugin_s_draws for why)

Run (once step4_neutral_counts.jsonl exists)::

    cd swap
    ../.venv-model/bin/python three_arm.py \
        --step3-counts ../artifacts/step3_counts.jsonl \
        --neutral-counts ../artifacts/step4_neutral_counts.jsonl \
        --baseline-counts ../artifacts/step4_baseline_counts.jsonl \
        --step3-summaries ../artifacts/step3_summaries.json \
        --out-dir ../artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from model import (
    DEFAULT_EPS,
    DEFAULT_GRID,
    DEFAULT_PROB,
    SwapData,
    commitment_summary,
    convergence,
    covertness_rate,
    dep_draws,
    fit,
    load_counts,
    plugin_s_draws,
    point_summary,
    population_fractions,
    prepare_data,
    prior_settings,
    replicate_constant_arm,
)

PAIRS = {
    "dep": ("orig", "swap"),
    "r": ("orig", "neutral"),
    "c": ("neutral", "swap"),
    "s": ("neutral", "baseline"),
}
INFORMED_DEP0 = {"dep": 0.62, "r": 0.31, "c": 0.31, "s": 0.0}
DEFAULT_EPS_STRIP = (0.1, 0.15, 0.2, 0.3)
POINT_T = 0.2


# --- record assembly ------------------------------------------------------


def merge_records(
    step3: list[dict],
    neutral: list[dict],
    baseline: list[dict],
    *,
    grid=DEFAULT_GRID,
) -> tuple[list[dict], dict]:
    """Merge the three counts files into one record list; report what was
    added and which neutral units have no baseline cell."""
    info: dict = {}
    step3_arms = {r["arm"] for r in step3}
    if not step3_arms <= {"orig", "swap"}:
        raise ValueError(f"step-3 counts carry unexpected arms {step3_arms}")
    bad = [r for r in neutral if r["arm"] != "neutral"]
    if bad:
        raise ValueError(f"{len(bad)} neutral-count records do not have arm == 'neutral'")
    bad = [r for r in baseline if r["arm"] != "baseline"]
    if bad:
        raise ValueError(f"{len(bad)} baseline records do not have arm == 'baseline'")

    constant_by_cell = {r["cell_id"]: (int(r["k"]), int(r["n"])) for r in baseline}
    info["n_baseline_cells"] = len(constant_by_cell)

    # t=0 cell of the neutral arm = the baseline record (empty prefix under
    # the no-bet prompt). Added only where the neutral file does not already
    # carry a t=0 cell-level record.
    have_t0 = {r["cell_id"] for r in neutral if r.get("is_cell_level") or r.get("t") == 0}
    neutral_t0 = [
        {**r, "arm": "neutral"} for r in baseline if r["cell_id"] not in have_t0
    ]
    info["n_neutral_t0_added"] = len(neutral_t0)

    neutral_all = neutral + neutral_t0
    replicas = replicate_constant_arm(
        neutral_all, constant_by_cell, arm_name="baseline", source_arm="neutral", grid=grid
    )
    info["n_baseline_replicas"] = len(replicas)
    missing_cells = sorted(
        {r["cell_id"] for r in neutral_all if r["cell_id"] not in constant_by_cell}
    )
    info["neutral_cells_without_baseline"] = missing_cells

    merged = list(step3) + neutral_all + replicas
    info["arms"] = dict(sorted(
        defaultdict(int, {a: sum(1 for r in merged if r["arm"] == a) for a in
                          {r["arm"] for r in merged}}).items()
    ))
    return merged, info


def parse_failure_table(records: list[dict], grid=DEFAULT_GRID) -> dict:
    """Unparsed share by arm x grid slot, from the count records' own
    ``n`` / ``n_unparsed`` fields (replicated baseline rows excluded)."""
    grid = np.asarray(grid, dtype=float)
    acc: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in records:
        if r.get("replicated"):
            continue
        g = int(np.argmin(np.abs(grid - float(r.get("frac_sentences", 0.0)))))
        slot = acc[(r["arm"], g)]
        slot[0] += int(r.get("n", 0))
        slot[1] += int(r.get("n_unparsed", 0))
        slot[2] += int(r.get("n_truncated", 0))
    out: dict = {}
    for (arm, g), (n, unparsed, trunc) in sorted(acc.items()):
        total = n + unparsed
        out.setdefault(arm, {})[str(float(grid[g]))] = {
            "n_parsed": n,
            "n_unparsed": unparsed,
            "unparsed_rate": (unparsed / total) if total else None,
            "n_truncated": trunc,
        }
    return out


def early_locked_units(step3_summaries_path: Path, prior: str = "neutral") -> list[str]:
    """Rollouts the step-3 fit never found dependent past t=0."""
    payload = json.loads(Path(step3_summaries_path).read_text())
    rows = payload[prior]["summaries"]
    return [s["unit_id"] for s in rows if s["is_rollout"] and not s["dependent_after_t0"]]


# --- shared-baseline / plug-in variants of s ---------------------------------

# Which tracked arm is p_neutral in each pair's fit.
NEUTRAL_VAR = {"r": "p_b", "c": "p_a", "s": "p_a"}


def shared_baseline_s_draws(idata, data: SwapData, constant_by_cell, *, pair: str, seed: int = 0):
    """s with p_neutral taken from `pair`'s fit and one shared Beta p_base
    draw per cell (see `model.plugin_s_draws`). For pair "s" this only
    replaces the replicated p_base by a shared draw; for "r" / "c" it is the
    plug-in estimand that avoids the constant-arm over-shrinkage."""
    return plugin_s_draws(idata, data, constant_by_cell, neutral_var=NEUTRAL_VAR[pair], seed=seed)


# --- one pair ---------------------------------------------------------------


def fit_pair(
    pair: str,
    merged: list[dict],
    *,
    priors,
    informed_dep0: float,
    eps_strip,
    early_locked: list[str],
    constant_by_cell: dict | None,
    fit_kwargs: dict,
    grid=DEFAULT_GRID,
) -> dict:
    arm_a, arm_b = PAIRS[pair]
    data = prepare_data(merged, grid=grid, arm_a=arm_a, arm_b=arm_b)
    print(
        f"\n=== pair {pair}: {arm_a} - {arm_b} | {data.n_units} units "
        f"({int(data.unit_is_rollout.sum())} rollouts), {data.n_cells} cells, "
        f"{len(data.obs_k)} binomial observations",
        flush=True,
    )
    n_a = int((data.obs_arm > 0).sum())
    n_b = int((data.obs_arm < 0).sum())
    print(f"    observations: {n_a} {arm_a}, {n_b} {arm_b}")
    settings = prior_settings(informed_dep0) if informed_dep0 != 0.0 else _settings_zero()
    results = {}
    for name in priors:
        t0 = time.time()
        idata = fit(data, settings[name], track_arms=True, **fit_kwargs)
        diag = convergence(idata)
        print(f"    {name:<10} {time.time() - t0:6.0f}s  {diag}", flush=True)
        per_eps = {}
        for eps in eps_strip:
            summaries = commitment_summary(idata, data, eps=eps)
            p_a = dep_draws(idata, "p_a").mean(axis=0)
            p_b = dep_draws(idata, "p_b").mean(axis=0)
            for u, s in enumerate(summaries):
                s[f"p_{arm_a}_mean"] = p_a[u].tolist()
                s[f"p_{arm_b}_mean"] = p_b[u].tolist()
            point = point_summary(idata, data, t=POINT_T, eps=eps)
            block = {
                "pair": {"name": pair, "arm_a": arm_a, "arm_b": arm_b},
                "eps": eps,
                "summaries": summaries,
                "covertness": covertness_rate(summaries),
                "convergence": diag,
                "point_t": POINT_T,
                "point": point,
                "early_locked": population_fractions(
                    idata, data, early_locked, t=POINT_T, eps=eps, prob=DEFAULT_PROB
                ),
            }
            if constant_by_cell is not None and pair in NEUTRAL_VAR:
                # s assembled from this fit's p_neutral and a shared p_base draw:
                # for pair "s" a check on the replicated-arm assumption; for
                # "r"/"c" the plug-in s that sidesteps the constant-arm
                # over-shrinkage (see model.plugin_s_draws).
                shared = shared_baseline_s_draws(idata, data, constant_by_cell, pair=pair)
                label = "shared_baseline" if pair == "s" else f"plugin_from_{pair}"
                block[f"early_locked_{label}"] = population_fractions(
                    idata, data, early_locked, t=POINT_T, eps=eps, prob=DEFAULT_PROB,
                    draws=shared,
                )
                block[f"point_s_{label}"] = point_summary(
                    idata, data, t=POINT_T, eps=eps, draws=shared
                )
                block["s_variant"] = label
            per_eps[eps] = block
        results[name] = {"idata": idata, "data": data, "per_eps": per_eps}
    return results


def _settings_zero():
    """Prior settings with the informed location at dep0 = 0 (s pair)."""
    from model import PriorSetting

    base = prior_settings(0.5)
    inf = base["informed"]
    base["informed"] = PriorSetting(
        name=inf.name, delta0_loc=0.0, delta0_scale=inf.delta0_scale,
        delta_step_scale=inf.delta_step_scale, delta_cell_scale=inf.delta_cell_scale,
        delta_unit_step_scale=inf.delta_unit_step_scale,
    )
    return base


def _pair_table(results: dict, pair: str) -> str:
    arm_a, arm_b = PAIRS[pair]
    lines = [
        f"{'prior':<11} {pair+'(0)':>8} {pair+'(.2)':>8} {pair+'(1)':>8} "
        f"{'%>eps past t0':>14} {'EL: P(>.2)>.9':>14} {'EL: P(|.|<.2)>.9':>17} {'Rhat':>6}"
    ]
    for name, res in results.items():
        block = res["per_eps"][DEFAULT_EPS] if DEFAULT_EPS in res["per_eps"] else next(iter(res["per_eps"].values()))
        data = res["data"]
        dep = dep_draws(res["idata"])
        rollouts = [i for i, r in enumerate(data.unit_is_rollout) if r]
        g2 = int(np.argmin(np.abs(data.grid - POINT_T)))
        el = block["early_locked"]
        lines.append(
            f"{name:<11} {dep[:, rollouts, 0].mean():>8.3f} {dep[:, rollouts, g2].mean():>8.3f} "
            f"{dep[:, rollouts, -1].mean():>8.3f} {block['covertness']['rate']:>13.1%} "
            f"{el.get('classified_gt_eps_frac', float('nan')):>14.3f} "
            f"{el.get('classified_abs_lt_eps_frac', float('nan')):>17.3f} "
            f"{block['convergence']['max_rhat']:>6.3f}"
        )
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------


def _parse_informed(spec: str | None) -> dict:
    out = dict(INFORMED_DEP0)
    if spec:
        for item in spec.split(","):
            k, v = item.split("=")
            out[k.strip()] = float(v)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step3-counts", type=Path, required=True)
    p.add_argument("--neutral-counts", type=Path, required=True)
    p.add_argument("--baseline-counts", type=Path, required=True)
    p.add_argument("--step3-summaries", type=Path, required=True)
    p.add_argument("--early-locked-prior", default="neutral")
    p.add_argument("--pairs", default="r,c,s", help="subset of dep,r,c,s")
    p.add_argument("--prior", default="all", choices=["all", "skeptical", "neutral", "informed"])
    p.add_argument("--informed-dep0", default=None, help="e.g. r=0.31,c=0.31,s=0")
    p.add_argument("--eps-strip", default=",".join(str(e) for e in DEFAULT_EPS_STRIP))
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nuts-sampler", default="pymc")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", default="step4")
    p.add_argument("--no-netcdf", action="store_true")
    args = p.parse_args(argv)

    pairs = [x.strip() for x in args.pairs.split(",") if x.strip()]
    unknown = [x for x in pairs if x not in PAIRS]
    if unknown:
        p.error(f"unknown pairs {unknown}; choose from {list(PAIRS)}")
    eps_strip = [float(e) for e in args.eps_strip.split(",")]
    if DEFAULT_EPS not in eps_strip:
        eps_strip.append(DEFAULT_EPS)
    priors = ("skeptical", "neutral", "informed") if args.prior == "all" else (args.prior,)
    informed = _parse_informed(args.informed_dep0)

    step3 = load_counts(args.step3_counts)
    neutral = load_counts(args.neutral_counts)
    baseline = load_counts(args.baseline_counts)
    merged, info = merge_records(step3, neutral, baseline)
    constant_by_cell = {r["cell_id"]: (int(r["k"]), int(r["n"])) for r in baseline}
    print(json.dumps(info, indent=2))
    if info["neutral_cells_without_baseline"]:
        print("WARNING: neutral cells without a baseline record (s undefined there):",
              info["neutral_cells_without_baseline"], file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.tag}_merged_counts.jsonl"
    with open(merged_path, "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    print(f"merged counts -> {merged_path}")

    parse_table = parse_failure_table(merged)
    print("\nunparsed rate by arm x grid slot:")
    for arm, row in parse_table.items():
        cells = "  ".join(
            f"{t}:{v['unparsed_rate']:.3f}" if v["unparsed_rate"] is not None else f"{t}:-"
            for t, v in row.items()
        )
        print(f"  {arm:<9} {cells}")

    early = early_locked_units(args.step3_summaries, args.early_locked_prior)
    print(f"\nearly-locked set ({args.early_locked_prior} prior, dependent_after_t0 == False): {len(early)} rollouts")

    fit_kwargs = dict(
        draws=args.draws, tune=args.tune, chains=args.chains,
        target_accept=args.target_accept, seed=args.seed, nuts_sampler=args.nuts_sampler,
    )
    early_locked_out: dict = {
        "early_locked_prior": args.early_locked_prior,
        "n_early_locked": len(early),
        "point_t": POINT_T,
        "prob": DEFAULT_PROB,
        "pairs": {},
    }
    for pair in pairs:
        results = fit_pair(
            pair, merged, priors=priors, informed_dep0=informed[pair], eps_strip=eps_strip,
            early_locked=early, constant_by_cell=constant_by_cell,
            fit_kwargs=fit_kwargs,
        )
        print(_pair_table(results, pair))
        for eps in eps_strip:
            payload = {name: res["per_eps"][eps] for name, res in results.items()}
            suffix = "" if eps == DEFAULT_EPS else f"_eps{eps:g}"
            path = args.out_dir / f"{args.tag}_summaries_{pair}{suffix}.json"
            path.write_text(json.dumps(payload, indent=2))
            print(f"  summaries -> {path}")
        early_locked_out["pairs"][pair] = {
            name: {
                f"eps{eps:g}": {
                    "pair_estimand": res["per_eps"][eps]["early_locked"],
                    **({
                        "s_" + res["per_eps"][eps]["s_variant"]:
                            res["per_eps"][eps]["early_locked_" + res["per_eps"][eps]["s_variant"]]
                    } if "s_variant" in res["per_eps"][eps] else {}),
                }
                for eps in eps_strip
            }
            for name, res in results.items()
        }
        if not args.no_netcdf:
            for name, res in results.items():
                path = args.out_dir / f"{args.tag}_idata_{pair}_{name}.nc"
                res["idata"].to_netcdf(str(path))
                print(f"  idata -> {path}")
        # free memory between pairs
        del results

    path = args.out_dir / f"{args.tag}_early_locked.json"
    early_locked_out["parse_failures"] = parse_table
    early_locked_out["merge_info"] = info
    path.write_text(json.dumps(early_locked_out, indent=2))
    print(f"\nearly-locked fractions -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
