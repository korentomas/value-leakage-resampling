"""Synthetic-data recovery test for `model.py`.

Generates rollouts with *known* commitment points and dep magnitudes, runs the
hierarchical model on the resulting binomial counts, and checks three things:

1. **Point recovery** — correlation and RMSE between posterior-mean dep and
   true dep across all (rollout, grid) cells.
2. **Calibration** — coverage of the 50% / 80% / 95% credible intervals. A
   well-behaved model puts the true value inside its 95% interval about 95%
   of the time; systematic under-coverage means the partial pooling is too
   aggressive for this design.
3. **Commitment recovery** — how close the estimated collapse position is to
   the simulated one, and whether rollouts that never commit are correctly
   left uncollapsed.

The generator is deliberately *not* the model's own generative process: true
dep follows a logistic decay in CoT fraction with a rollout-specific
commitment point, while the model uses a Gaussian random walk over the grid.
Recovery under that mismatch is the thing worth believing.

Run: ``python synth_test.py`` (add ``--quick`` for a smoke test).

Three-arm mode (step 4A): ``python synth_test.py --three-arm`` generates a
neutral arm and a per-question baseline with known s/r/c curves
(``s = p_neu - p_base``, ``r = p_orig - p_neu``, ``c = p_neu - p_swap``),
fits each pair with the unchanged model via `model.prepare_data(arm_a=,
arm_b=)` (the s pair through `replicate_constant_arm`), and reports
recovery, calibration and the s(0.2) classifications against the truth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from model import (
    DEFAULT_EPS,
    DEFAULT_GRID,
    DEFAULT_PROB,
    commitment_summary,
    convergence,
    crossing_fraction,
    dep_draws,
    fit,
    plugin_s_draws,
    point_summary,
    prepare_data,
    prior_settings,
    replicate_constant_arm,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def delta_from_dep(dep: float, alpha: float, *, tol: float = 1e-10) -> float:
    """Solve sigmoid(alpha + d/2) - sigmoid(alpha - d/2) = dep for d.

    The left side is odd in d and strictly increasing, so bisection on
    [0, hi] with a sign flip covers every attainable dep in (-1, 1).
    """
    if dep == 0.0:
        return 0.0
    sign = 1.0 if dep > 0 else -1.0
    target = abs(dep)
    lo, hi = 0.0, 2.0
    while sigmoid(alpha + hi / 2) - sigmoid(alpha - hi / 2) < target:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError(f"dep={dep} unattainable at alpha={alpha}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sigmoid(alpha + mid / 2) - sigmoid(alpha - mid / 2) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return sign * 0.5 * (lo + hi)


@dataclass
class Truth:
    grid: np.ndarray
    unit_ids: list[str]
    dep: np.ndarray             # (n_rollouts, n_grid) true dep
    commitment: np.ndarray      # (n_rollouts,) true commitment fraction
    committed: np.ndarray       # (n_rollouts,) bool: does it ever collapse
    cell_dep0: dict[str, float]


def simulate(
    *,
    n_rollouts: int = 50,
    n_questions: int = 5,
    grid=DEFAULT_GRID,
    n_continuations: int = 20,
    n_t0: int = 100,
    dep0_mean: float = 0.55,
    dep0_sd: float = 0.12,
    decay_width: float = 0.12,
    frac_never_commit: float = 0.25,
    alpha_sd: float = 0.6,
    alpha_walk_sd: float = 0.25,
    seed: int = 0,
):
    """Build synthetic `counts.jsonl` records plus the ground truth.

    Each rollout gets a commitment fraction ``m``; its true dep curve is

        dep(f) = dep0_cell * sigmoid((m - f) / width) / sigmoid(m / width)

    which equals ``dep0_cell`` exactly at ``f = 0`` — matching the identity
    that dep(0) is a cell-level quantity, not a rollout-level one — and
    decays through zero around ``f = m``. A configurable share of rollouts
    are given ``m > 1`` so they never collapse inside the observed range;
    those test the model's ability to *not* report a spurious commitment.
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid, dtype=float)
    n_g = len(grid)

    questions = [f"v1_q{q}" for q in range(n_questions)]
    directions = ["below_good", "above_good"]
    cell_dep0 = {}
    for q in questions:
        # dep(0) is a property of the question (it is the balanced bias
        # score), so both directions of a question share it.
        d0 = float(np.clip(rng.normal(dep0_mean, dep0_sd), 0.05, 0.9))
        for d in directions:
            cell_dep0[f"{q}|{d}|1000"] = d0

    cell_ids = list(cell_dep0)
    records = []

    # --- t = 0 cell-level observations (in the real pipeline these come
    # free from the paper's cached rollouts) ---
    for cell_id in cell_ids:
        q, direction, thr = cell_id.split("|")
        d0 = cell_dep0[cell_id]
        alpha0 = float(rng.normal(0.0, alpha_sd))
        delta0 = delta_from_dep(d0, alpha0)
        for arm, sign in (("orig", 1.0), ("swap", -1.0)):
            p = sigmoid(alpha0 + sign * delta0 / 2.0)
            records.append(
                {
                    "unit_id": cell_id,
                    "is_cell_level": True,
                    "cell_id": cell_id,
                    "prompt_key": q,
                    "direction": direction,
                    "threshold": float(thr),
                    "t": 0,
                    "arm": arm,
                    "frac_sentences": 0.0,
                    "k": int(rng.binomial(n_t0, p)),
                    "n": n_t0,
                }
            )

    # --- rollout-level observations at f > 0 ---
    unit_ids, dep_true, commitments, committed = [], [], [], []
    for i in range(n_rollouts):
        cell_id = cell_ids[i % len(cell_ids)]
        q, direction, thr = cell_id.split("|")
        d0 = cell_dep0[cell_id]
        never = rng.random() < frac_never_commit
        m = float(rng.uniform(1.15, 1.6)) if never else float(rng.uniform(0.05, 0.95))
        decay = sigmoid((m - grid) / decay_width) / sigmoid(m / decay_width)
        dep_curve = d0 * decay

        alpha = np.cumsum(
            np.concatenate([[rng.normal(0.0, alpha_sd)],
                            rng.normal(0.0, alpha_walk_sd, n_g - 1)])
        )
        unit_id = f"r{i:04d}"
        for g, frac in enumerate(grid):
            if g == 0:
                continue  # t=0 is the shared cell-level observation
            delta = delta_from_dep(float(dep_curve[g]), float(alpha[g]))
            for arm, sign in (("orig", 1.0), ("swap", -1.0)):
                p = sigmoid(alpha[g] + sign * delta / 2.0)
                records.append(
                    {
                        "unit_id": unit_id,
                        "is_cell_level": False,
                        "cell_id": cell_id,
                        "prompt_key": q,
                        "direction": direction,
                        "threshold": float(thr),
                        "t": g,
                        "arm": arm,
                        "frac_sentences": float(frac),
                        "k": int(rng.binomial(n_continuations, p)),
                        "n": n_continuations,
                    }
                )
        unit_ids.append(unit_id)
        dep_true.append(dep_curve)
        commitments.append(m)
        committed.append(not never)

    truth = Truth(
        grid=grid,
        unit_ids=unit_ids,
        dep=np.array(dep_true),
        commitment=np.array(commitments),
        committed=np.array(committed, dtype=bool),
        cell_dep0=cell_dep0,
    )
    return records, truth


def evaluate(idata, data, truth: Truth, *, eps: float = DEFAULT_EPS) -> dict:
    """Recovery, calibration, and commitment-point metrics."""
    dep = dep_draws(idata)  # (draws, units, grid)
    index = {u: i for i, u in enumerate(data.unit_ids)}
    rows = [index[u] for u in truth.unit_ids]
    post = dep[:, rows, :]                     # (draws, n_rollouts, n_grid)
    true = truth.dep                            # (n_rollouts, n_grid)

    mean = post.mean(axis=0)
    resid = mean - true
    # t=0 is an identity shared with the cell, so it is reported separately
    # from the rollout-level positions the model actually has to learn.
    interior = slice(1, None)

    def coverage(level):
        half = (100.0 - level) / 2.0
        lo = np.percentile(post, half, axis=0)
        hi = np.percentile(post, 100.0 - half, axis=0)
        inside = (true >= lo) & (true <= hi)
        return {
            "all": float(inside.mean()),
            "interior": float(inside[:, interior].mean()),
            "width_mean": float((hi - lo)[:, interior].mean()),
        }

    summaries = commitment_summary(idata, data, eps=eps)
    by_id = {s["unit_id"]: s for s in summaries}

    # Compare like with like: the model estimates where the *dep curve*
    # crosses eps, so score it against the true curve's crossing on the same
    # grid, not against the generator's latent commitment parameter (the two
    # differ by a fixed offset set by the decay width).
    true_cross = crossing_fraction(true, truth.grid, eps)

    est_collapse, true_collapse = [], []
    false_collapse = missed_collapse = 0
    cross_est, cross_true, cross_inside = [], [], []
    for i, unit_id in enumerate(truth.unit_ids):
        s = by_id[unit_id]
        crosses = not np.isnan(true_cross[i])
        if crosses:
            if s["t_collapse"] is None:
                missed_collapse += 1
            else:
                est_collapse.append(s["t_collapse"])
                true_collapse.append(true_cross[i])
            if s["commitment_frac"] is not None:
                cross_est.append(s["commitment_frac"])
                cross_true.append(true_cross[i])
                cross_inside.append(
                    s["commitment_lo95"] <= true_cross[i] <= s["commitment_hi95"]
                )
        elif s["t_collapse"] is not None:
            false_collapse += 1

    def _corr_mae(est, tru):
        est, tru = np.asarray(est), np.asarray(tru)
        if len(est) < 3:
            return float("nan"), float("nan")
        return (
            float(np.corrcoef(est, tru)[0, 1]),
            float(np.abs(est - tru).mean()),
        )

    thresh_corr, thresh_mae = _corr_mae(est_collapse, true_collapse)
    cont_corr, cont_mae = _corr_mae(cross_est, cross_true)

    return {
        "commitment_threshold_corr": thresh_corr,
        "commitment_threshold_mae": thresh_mae,
        "commitment_corr": cont_corr,
        "commitment_mae": cont_mae,
        "commitment_coverage_95": (
            float(np.mean(cross_inside)) if cross_inside else float("nan")
        ),
        "n_crossing": int(np.isfinite(true_cross).sum()),
        "n_localised": len(cross_est),
        "n_rollouts": len(truth.unit_ids),
        "dep_corr": float(np.corrcoef(mean.ravel(), true.ravel())[0, 1]),
        "dep_corr_interior": float(
            np.corrcoef(mean[:, interior].ravel(), true[:, interior].ravel())[0, 1]
        ),
        "dep_rmse": float(np.sqrt((resid**2).mean())),
        "dep_rmse_interior": float(np.sqrt((resid[:, interior] ** 2).mean())),
        "dep_bias": float(resid[:, interior].mean()),
        "coverage_50": coverage(50.0),
        "coverage_80": coverage(80.0),
        "coverage_95": coverage(95.0),
        "n_committed": int(truth.committed.sum()),
        "missed_collapse": missed_collapse,
        "false_collapse": false_collapse,
        "convergence": convergence(idata),
    }


def _fmt(metrics: dict) -> str:
    c95, c80, c50 = (
        metrics["coverage_95"], metrics["coverage_80"], metrics["coverage_50"]
    )
    return "\n".join(
        [
            f"  rollouts               {metrics['n_rollouts']}",
            f"  dep corr (all/interior) {metrics['dep_corr']:.3f} / "
            f"{metrics['dep_corr_interior']:.3f}",
            f"  dep RMSE (all/interior) {metrics['dep_rmse']:.3f} / "
            f"{metrics['dep_rmse_interior']:.3f}",
            f"  dep bias (interior)     {metrics['dep_bias']:+.3f}",
            f"  coverage 95/80/50       {c95['interior']:.3f} / "
            f"{c80['interior']:.3f} / {c50['interior']:.3f}"
            f"   (95% mean width {c95['width_mean']:.3f})",
            f"  commitment corr / MAE   {metrics['commitment_corr']:.3f} / "
            f"{metrics['commitment_mae']:.3f}"
            f"   (95% coverage {metrics['commitment_coverage_95']:.3f}, "
            f"{metrics['n_localised']}/{metrics['n_crossing']} localised)",
            f"  threshold-test corr/MAE {metrics['commitment_threshold_corr']:.3f} / "
            f"{metrics['commitment_threshold_mae']:.3f}"
            f"   ({metrics['missed_collapse']} censored, "
            f"{metrics['false_collapse']} false)",
            f"  max Rhat {metrics['convergence']['max_rhat']:.3f}, "
            f"min ESS {metrics['convergence']['min_ess_bulk']:.0f}, "
            f"divergences {metrics['convergence']['n_divergent']}",
        ]
    )


# --- three-arm (step 4A) ----------------------------------------------------


@dataclass
class ThreeArmTruth:
    grid: np.ndarray
    unit_ids: list[str]
    curves: dict          # {"s": (n_rollouts, n_grid), "r": ..., "c": ...}
    p_base: dict          # cell_id -> p_base on that direction's favoured side
    bet_insensitive: np.ndarray   # (n_rollouts,) bool: s == 0 by construction
    committed: np.ndarray


def simulate_three_arm(
    *,
    n_rollouts: int = 50,
    n_questions: int = 5,
    grid=DEFAULT_GRID,
    n_continuations: int = 25,
    n_t0: int = 100,
    n_base: int = 100,
    dep0_mean: float = 0.55,
    dep0_sd: float = 0.12,
    decay_width: float = 0.12,
    frac_never_commit: float = 0.25,
    frac_bet_insensitive: float = 0.3,
    s_max_mean: float = 0.35,
    s_max_sd: float = 0.1,
    seed: int = 0,
):
    """Synthetic three-arm counts with known s / r / c curves.

    Per question: ``p_base`` (the no-bet favoured-side rate, ~0.5 as in the
    real design), ``dep0`` split into ``r0 + c0`` with a random share.
    Per rollout, with ``decay(f)`` the same logistic commitment profile as
    the two-arm generator (``decay(0) = 1``):

        r(f) = r0 * decay(f)                      prompt pull fades as the CoT commits
        c(f) = c0 * decay(f)
        s(f) = s_max * (1 - decay(f))             prefix-encoded steering rises
                                                  (0 for the bet-insensitive share)
        p_neu = p_base + s,  p_orig = p_neu + r,  p_swap = p_neu - c

    all clipped to [0.02, 0.98], after which the *true* s / r / c are
    recomputed from the clipped probabilities so the truth is exactly what
    the model is asked to recover. Not the model's generative process: the
    model fits each pair with an independent random walk, and the three
    pairs share nothing.

    t=0: orig/swap cells with ``n_t0`` draws; the neutral t=0 cell *is* the
    baseline draw (``n_base``), as in the real pipeline; baseline replicas
    come from `replicate_constant_arm`, the production path.
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid, dtype=float)
    n_g = len(grid)
    directions = ["below_good", "above_good"]
    questions = [f"v1_q{q}" for q in range(n_questions)]

    cell_par, p_base, records = {}, {}, []
    for q in questions:
        d0 = float(np.clip(rng.normal(dep0_mean, dep0_sd), 0.05, 0.9))
        share = float(rng.uniform(0.3, 0.7))
        pb_below = float(rng.uniform(0.45, 0.55))
        k_below = int(rng.binomial(n_base, pb_below))
        for d in directions:
            cell_id = f"{q}|{d}|1000"
            pb = pb_below if d == "below_good" else 1.0 - pb_below
            kb = k_below if d == "below_good" else n_base - k_below
            cell_par[cell_id] = {"r0": share * d0, "c0": (1 - share) * d0, "p_base": pb}
            p_base[cell_id] = pb
            # baseline record (constant arm) and neutral t=0 cell (same draw)
            for arm in ("baseline", "neutral"):
                records.append(
                    {
                        "unit_id": cell_id, "is_cell_level": True, "cell_id": cell_id,
                        "prompt_key": q, "direction": d, "threshold": 1000.0, "t": 0,
                        "arm": arm, "frac_sentences": 0.0, "k": kb, "n": n_base,
                    }
                )
            p_neu0 = pb
            p_orig0 = float(np.clip(p_neu0 + share * d0, 0.02, 0.98))
            p_swap0 = float(np.clip(p_neu0 - (1 - share) * d0, 0.02, 0.98))
            for arm, p in (("orig", p_orig0), ("swap", p_swap0)):
                records.append(
                    {
                        "unit_id": cell_id, "is_cell_level": True, "cell_id": cell_id,
                        "prompt_key": q, "direction": d, "threshold": 1000.0, "t": 0,
                        "arm": arm, "frac_sentences": 0.0,
                        "k": int(rng.binomial(n_t0, p)), "n": n_t0,
                    }
                )

    cell_ids = list(cell_par)
    unit_ids, S, R, C, insens, committed = [], [], [], [], [], []
    for i in range(n_rollouts):
        cell_id = cell_ids[i % len(cell_ids)]
        q, d, _ = cell_id.split("|")
        par = cell_par[cell_id]
        never = rng.random() < frac_never_commit
        m = float(rng.uniform(1.15, 1.6)) if never else float(rng.uniform(0.05, 0.95))
        decay = sigmoid((m - grid) / decay_width) / sigmoid(m / decay_width)
        bet_insens = rng.random() < frac_bet_insensitive
        s_max = 0.0 if bet_insens else float(np.clip(rng.normal(s_max_mean, s_max_sd), 0.0, 0.45))
        p_neu = np.clip(par["p_base"] + s_max * (1 - decay), 0.02, 0.98)
        p_orig = np.clip(p_neu + par["r0"] * decay, 0.02, 0.98)
        p_swap = np.clip(p_neu - par["c0"] * decay, 0.02, 0.98)
        unit_id = f"r{i:04d}"
        for g, frac in enumerate(grid):
            if g == 0:
                continue
            for arm, p in (("orig", p_orig[g]), ("swap", p_swap[g]), ("neutral", p_neu[g])):
                records.append(
                    {
                        "unit_id": unit_id, "is_cell_level": False, "cell_id": cell_id,
                        "prompt_key": q, "direction": d, "threshold": 1000.0, "t": g,
                        "arm": arm, "frac_sentences": float(frac),
                        "k": int(rng.binomial(n_continuations, p)), "n": n_continuations,
                    }
                )
        unit_ids.append(unit_id)
        S.append(p_neu - par["p_base"])
        R.append(p_orig - p_neu)
        C.append(p_neu - p_swap)
        insens.append(bet_insens)
        committed.append(not never)

    constant = {
        r["cell_id"]: (r["k"], r["n"]) for r in records if r["arm"] == "baseline"
    }
    records += replicate_constant_arm(records, constant, arm_name="baseline", source_arm="neutral", grid=grid)
    truth = ThreeArmTruth(
        grid=grid, unit_ids=unit_ids,
        curves={"s": np.array(S), "r": np.array(R), "c": np.array(C)},
        p_base=p_base, bet_insensitive=np.array(insens, dtype=bool),
        committed=np.array(committed, dtype=bool),
    )
    return records, truth


THREE_ARM_PAIRS = {"r": ("orig", "neutral"), "c": ("neutral", "swap"), "s": ("neutral", "baseline")}


def evaluate_pair(
    idata, data, truth: ThreeArmTruth, pair: str, *, eps: float = DEFAULT_EPS,
    draws=None, label: str | None = None,
) -> dict:
    """Recovery + calibration of one pair's curve; s adds the 0.2 readings.
    `draws` overrides the posterior (used for the plug-in s variants)."""
    dep = dep_draws(idata) if draws is None else np.asarray(draws)
    index = {u: i for i, u in enumerate(data.unit_ids)}
    rows = [index[u] for u in truth.unit_ids]
    post = dep[:, rows, :]
    true = truth.curves[pair]
    mean = post.mean(axis=0)
    resid = mean - true
    interior = slice(1, None)

    def coverage(level):
        half = (100.0 - level) / 2.0
        lo = np.percentile(post, half, axis=0)
        hi = np.percentile(post, 100.0 - half, axis=0)
        inside = (true >= lo) & (true <= hi)
        return {
            "interior": float(inside[:, interior].mean()),
            "width_mean": float((hi - lo)[:, interior].mean()),
        }

    out = {
        "pair": label or pair,
        "corr_interior": float(np.corrcoef(mean[:, interior].ravel(), true[:, interior].ravel())[0, 1]),
        "rmse_interior": float(np.sqrt((resid[:, interior] ** 2).mean())),
        "bias_interior": float(resid[:, interior].mean()),
        "t0_rmse": float(np.sqrt((resid[:, 0] ** 2).mean())),
        "coverage_50": coverage(50.0),
        "coverage_80": coverage(80.0),
        "coverage_95": coverage(95.0),
        "convergence": convergence(idata),
    }
    if pair == "s":
        g = int(np.argmin(np.abs(truth.grid - 0.2)))
        pts = {p["unit_id"]: p for p in point_summary(idata, data, t=0.2, eps=eps, draws=draws)}
        true_gt = true[:, g] > eps
        true_lt = np.abs(true[:, g]) < eps
        cls_gt = np.array([pts[u]["p_gt_eps"] > DEFAULT_PROB for u in truth.unit_ids])
        cls_lt = np.array([pts[u]["p_abs_lt_eps"] > DEFAULT_PROB for u in truth.unit_ids])

        def pr(cls, tru):
            tp = int((cls & tru).sum())
            return {
                "n_true": int(tru.sum()), "n_classified": int(cls.sum()),
                "precision": tp / cls.sum() if cls.sum() else float("nan"),
                "recall": tp / tru.sum() if tru.sum() else float("nan"),
            }

        out["s02_gt_eps"] = pr(cls_gt, true_gt)
        out["s02_abs_lt_eps"] = pr(cls_lt, true_lt)
        out["s02_bet_insensitive_classified_lt_eps"] = float(
            cls_lt[truth.bet_insensitive].mean()
        ) if truth.bet_insensitive.any() else float("nan")
        out["s02_insensitive_false_gt"] = int(cls_gt[truth.bet_insensitive].sum())
    return out


def _fmt_pair(m: dict) -> str:
    lines = [
        f"  {m['pair']}: corr {m['corr_interior']:.3f}  RMSE {m['rmse_interior']:.3f}  "
        f"bias {m['bias_interior']:+.3f}  t0 RMSE {m['t0_rmse']:.3f}  "
        f"coverage 95/80/50 {m['coverage_95']['interior']:.3f} / "
        f"{m['coverage_80']['interior']:.3f} / {m['coverage_50']['interior']:.3f}  "
        f"(95% width {m['coverage_95']['width_mean']:.3f})  "
        f"Rhat {m['convergence']['max_rhat']:.3f} ESS {m['convergence']['min_ess_bulk']:.0f} "
        f"div {m['convergence']['n_divergent']}"
    ]
    if "s02_gt_eps" in m:
        a, b = m["s02_gt_eps"], m["s02_abs_lt_eps"]
        lines.append(
            f"     s(0.2) > eps: {a['n_classified']} classified / {a['n_true']} true, "
            f"precision {a['precision']:.3f} recall {a['recall']:.3f}; "
            f"|s(0.2)| < eps: {b['n_classified']} / {b['n_true']}, "
            f"precision {b['precision']:.3f} recall {b['recall']:.3f}; "
            f"bet-insensitive units classified |s|<eps: "
            f"{m['s02_bet_insensitive_classified_lt_eps']:.3f}, "
            f"falsely >eps: {m['s02_insensitive_false_gt']}"
        )
    return "\n".join(lines)


def main_three_arm(args) -> int:
    records, truth = simulate_three_arm(
        n_rollouts=args.n_rollouts, n_questions=args.n_questions,
        n_continuations=args.n_continuations, seed=args.seed,
    )
    which = ("skeptical", "neutral", "informed") if args.prior == "all" else (args.prior,)
    # Informed locations follow three_arm.py: r, c at half the published bias; s at 0.
    informed = {"r": 0.31, "c": 0.31, "s": 0.0}
    constant = {
        r["cell_id"]: (r["k"], r["n"]) for r in records
        if r["arm"] == "baseline" and not r.get("replicated")
    }
    out = {}
    for pair, (arm_a, arm_b) in THREE_ARM_PAIRS.items():
        data = prepare_data(records, arm_a=arm_a, arm_b=arm_b)
        print(
            f"\n=== pair {pair} = p[{arm_a}] - p[{arm_b}]: {data.n_units} units, "
            f"{len(data.obs_k)} observations ==="
        )
        for name in which:
            settings = prior_settings(informed[pair]) if informed[pair] > 0 else prior_settings()
            if informed[pair] == 0 and name == "informed":
                from model import PriorSetting

                inf = settings["informed"]
                settings["informed"] = PriorSetting(
                    name="informed", delta0_loc=0.0, delta0_scale=inf.delta0_scale,
                    delta_step_scale=inf.delta_step_scale, delta_cell_scale=inf.delta_cell_scale,
                    delta_unit_step_scale=inf.delta_unit_step_scale,
                )
            idata = fit(
                data, settings[name], draws=args.draws, tune=args.tune, chains=args.chains,
                target_accept=args.target_accept, seed=args.seed, nuts_sampler=args.nuts_sampler,
                track_arms=True,
            )
            m = evaluate_pair(idata, data, truth, pair, eps=args.eps)
            out[f"{pair}/{name}"] = m
            print(f" prior {name}")
            print(_fmt_pair(m))
            # s assembled from this fit's p_neutral and a shared Beta p_base draw
            # (model.plugin_s_draws): for r / c the plug-in estimand, for s the
            # shared-baseline check on the replicated-arm assumption.
            neutral_var = {"r": "p_b", "c": "p_a", "s": "p_a"}[pair]
            sd = plugin_s_draws(idata, data, constant, neutral_var=neutral_var, seed=args.seed)
            label = f"s_plugin_from_{pair}" if pair != "s" else "s_shared_baseline"
            ms = evaluate_pair(idata, data, truth, "s", eps=args.eps, draws=sd, label=label)
            out[f"{label}/{name}"] = ms
            print(_fmt_pair(ms))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nmetrics -> {args.out}")
    gate_prior = "neutral" if "s/neutral" in out else which[0]
    checks = {}
    # r and c each carry about half of dep's range at the same N, so their
    # correlation floor is lower than the two-arm 0.85; RMSE/coverage bind.
    # The replicated-arm s is reported but not gated on correlation/coverage:
    # the constant arm over-shrinks it (see model.plugin_s_draws); the
    # plug-in s from the r fit is the gated version.
    for pair in ("r", "c", "s", "s_plugin_from_r", "s_plugin_from_c"):
        m = out[f"{pair}/{gate_prior}"]
        if pair in ("r", "c"):
            checks[f"{pair}: corr > 0.80"] = m["corr_interior"] > 0.80
        if pair.startswith("s_plugin"):
            checks[f"{pair}: corr > 0.80"] = m["corr_interior"] > 0.80
            checks[f"{pair}: 95% coverage in [0.90, 0.99]"] = 0.90 <= m["coverage_95"]["interior"] <= 0.99
        checks[f"{pair}: |bias| < 0.03"] = abs(m["bias_interior"]) < 0.03
        if pair in ("r", "c"):
            checks[f"{pair}: 95% coverage in [0.90, 0.99]"] = 0.90 <= m["coverage_95"]["interior"] <= 0.99
        checks[f"{pair}: max Rhat < 1.02"] = m["convergence"]["max_rhat"] < 1.02
        checks[f"{pair}: divergences == 0"] = m["convergence"]["n_divergent"] == 0
    for pair in ("s", "s_plugin_from_r"):
        ms = out[f"{pair}/{gate_prior}"]
        a, b = ms["s02_gt_eps"], ms["s02_abs_lt_eps"]
        if a["n_classified"] > 0:
            checks[f"{pair}: s(0.2) > eps precision > 0.9"] = a["precision"] > 0.9
        checks[f"{pair}: s(0.2) |.| < eps precision > 0.9"] = b["precision"] > 0.9
        checks[f"{pair}: no bet-insensitive unit classified s(0.2) > eps"] = ms["s02_insensitive_false_gt"] == 0
    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = all(checks.values())
    print(f"\nthree-arm recovery gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-arm", action="store_true", help="step-4A s/r/c recovery")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--n-continuations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--nuts-sampler", default="pymc")
    parser.add_argument(
        "--prior", default="all", choices=["all", "skeptical", "neutral", "informed"]
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.quick:
        args.n_rollouts, args.draws, args.tune, args.chains = 12, 300, 300, 2

    if args.three_arm:
        return main_three_arm(args)

    records, truth = simulate(
        n_rollouts=args.n_rollouts,
        n_questions=args.n_questions,
        n_continuations=args.n_continuations,
        seed=args.seed,
    )
    data = prepare_data(records)
    print(
        f"simulated {args.n_rollouts} rollouts over {data.n_cells} cells; "
        f"{data.n_units} units, {len(data.obs_k)} binomial observations, "
        f"{int(truth.committed.sum())} commit inside the grid"
    )

    which = (
        ("skeptical", "neutral", "informed") if args.prior == "all" else (args.prior,)
    )
    settings = prior_settings()
    out = {}
    for name in which:
        print(f"\n=== prior: {name} ===")
        idata = fit(
            data,
            settings[name],
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            seed=args.seed,
            nuts_sampler=args.nuts_sampler,
        )
        metrics = evaluate(idata, data, truth, eps=args.eps)
        out[name] = metrics
        print(_fmt(metrics))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nmetrics -> {args.out}")

    # Pass/fail gate on the neutral prior: the run is only useful if the
    # model recovers dep at all and its intervals are not badly optimistic.
    gate = out.get("neutral") or next(iter(out.values()))
    checks = {
        "dep_corr_interior > 0.85": gate["dep_corr_interior"] > 0.85,
        "|dep_bias| < 0.02": abs(gate["dep_bias"]) < 0.02,
        "95% coverage in [0.90, 0.99]": 0.90 <= gate["coverage_95"]["interior"] <= 0.99,
        "80% coverage in [0.72, 0.90]": 0.72 <= gate["coverage_80"]["interior"] <= 0.90,
        "commitment corr > 0.75": gate["commitment_corr"] > 0.75,
        "commitment 95% coverage > 0.85": gate["commitment_coverage_95"] > 0.85,
        "max Rhat < 1.02": gate["convergence"]["max_rhat"] < 1.02,
        "divergences == 0": gate["convergence"]["n_divergent"] == 0,
    }
    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = all(checks.values())
    print(f"\nrecovery gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
