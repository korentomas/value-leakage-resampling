"""Hierarchical Bayesian model for per-rollout counterfactual dependence.

Estimand
--------
For source rollout *i* and cut position *t*,

    dep_i(t) = P(favoured side | original prompt, prefix_i(t))
             - P(favoured side | swapped prompt, prefix_i(t))

observed as two binomial counts, ``k_orig / n_orig`` and ``k_swap / n_swap``.

Position axis
-------------
Cuts are placed at sentence boundaries, and rollouts have different sentence
counts (and jittered cut positions), so ``t`` is not comparable across
rollouts as an integer. The model works on **fraction of the CoT consumed**,
binned to a common grid (default 0, .2, .4, .6, .8, 1). Each observation is
assigned to its nearest grid point; `driver.aggregate_counts` already carries
`frac_sentences`.

Structure
---------
On the logit scale, with *u* a rollout, *g* a grid position:

    logit p_orig[u,g] = alpha[u,g] + delta[u,g]/2
    logit p_swap[u,g] = alpha[u,g] - delta[u,g]/2
    dep[u,g]          = sigmoid(alpha + delta/2) - sigmoid(alpha - delta/2)

`alpha` is a nuisance (how favourable the prefix is regardless of prompt);
`delta` is the log-odds dependence. Both decompose as

    delta[u,g] = gamma[g] + nu[c(u),g] + w[u,g]

with `gamma` a global random walk over g, `nu` a (question x direction) cell
deviation — this is the partial pooling across rollouts within a cell — and
`w[u,·]` a per-rollout random walk **anchored at w[u,0] = 0**.

That anchor is not a convenience. At ``t = 0`` the prefix is empty, so the
two arms' requests do not depend on the rollout at all: every rollout in a
cell has literally the same dep(0), which equals the cell's balanced bias
score (see `driver.py`). Forcing w[u,0] = 0 encodes an identity, not an
assumption, and it means rollouts can only diverge as CoT accumulates —
exactly the commitment structure we want to measure.

Prior settings
--------------
Three settings, switchable with `--prior`, reported side by side. The
project's binding rule is that TK's own priors may set variance scales and
instrument design but never effect locations, so only the `informed` setting
carries a location, and that location is derived from the paper's *published*
balanced-bias figure (0.62 for Qwen3.5-35B-A3B), converted to the logit scale
by the exact identity dep = tanh(delta/4) at alpha = 0.

  skeptical  dep(0) prior concentrated near 0; tight walk scales.
  neutral    wide, location-free.
  informed   dep(0) centred on the paper's published bias; scales as neutral.

If a confirmatory conclusion flips across the three, the project rule is to
report "data insufficient" rather than pick one.

Arm pairs (step 4A)
-------------------
The model only ever sees *two* arms; which two is a property of the data
preparation, not of the model. `prepare_data(records, arm_a=, arm_b=)`
keeps the records whose ``arm`` is one of the pair, codes ``arm_a`` as +1
and ``arm_b`` as -1, and drops everything else, so the same model yields

    (orig, swap)        dep = p_orig - p_swap          (step 3, the default)
    (orig, neutral)     r   = p_orig - p_neutral
    (neutral, swap)     c   = p_neutral - p_swap
    (neutral, baseline) s   = p_neutral - p_base

from one counts file that carries all arms. The last pair needs a
"baseline" arm that does not vary with the prefix: `replicate_constant_arm`
writes the question's cached baseline count into every (unit, grid slot)
that has a neutral observation. Assumption, stated once here and again in
the helper: p_base is a property of the question, shared by every unit in
the same (prompt_key, direction) cell, and the replication gives each unit
its own copy of that count, so the p_base uncertainty enters each unit's
s posterior at the right scale (n ~ 100) but is *not* correlated across
units. The three-arm CLI (`three_arm.py`) reports a shared-draw variant
next to it as a check.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_EPS = 0.2
DEFAULT_PROB = 0.9


# --- prior settings -------------------------------------------------------


@dataclass(frozen=True)
class PriorSetting:
    name: str
    delta0_loc: float          # prior mean of the global dep(0), logit scale
    delta0_scale: float
    delta_step_scale: float    # HalfNormal scale for the global walk step sd
    delta_cell_scale: float    # HalfNormal scale for cell-deviation sd
    delta_unit_step_scale: float  # HalfNormal scale for per-rollout walk step sd
    alpha0_scale: float = 2.0
    alpha_step_scale: float = 1.0
    alpha_cell_scale: float = 1.0
    alpha_unit_step_scale: float = 1.0


def dep_to_logit(dep: float) -> float:
    """delta such that dep = sigmoid(delta/2) - sigmoid(-delta/2) at alpha = 0.

    That difference is exactly ``tanh(delta/4)``, so ``delta = 4 atanh(dep)``.
    """
    if not -1.0 < dep < 1.0:
        raise ValueError("dep must lie strictly inside (-1, 1)")
    return 4.0 * math.atanh(dep)


def prior_settings(informed_dep0: float = 0.62) -> dict[str, PriorSetting]:
    """The three settings. `informed_dep0` is the paper's published bias."""
    return {
        "skeptical": PriorSetting(
            name="skeptical",
            delta0_loc=0.0,
            delta0_scale=0.35,
            delta_step_scale=0.25,
            delta_cell_scale=0.25,
            delta_unit_step_scale=0.25,
        ),
        "neutral": PriorSetting(
            name="neutral",
            delta0_loc=0.0,
            delta0_scale=2.5,
            delta_step_scale=1.0,
            delta_cell_scale=0.75,
            delta_unit_step_scale=0.75,
        ),
        "informed": PriorSetting(
            name="informed",
            delta0_loc=dep_to_logit(informed_dep0),
            delta0_scale=0.75,
            delta_step_scale=1.0,
            delta_cell_scale=0.75,
            delta_unit_step_scale=0.75,
        ),
    }


# --- data preparation -----------------------------------------------------


@dataclass
class SwapData:
    """Design matrices for the model, built from `counts.jsonl` records."""

    unit_ids: list[str]
    cell_ids: list[str]
    grid: np.ndarray
    unit_cell: np.ndarray        # (n_units,) cell index per unit
    unit_is_rollout: np.ndarray  # (n_units,) bool
    obs_unit: np.ndarray         # (n_obs,)
    obs_grid: np.ndarray         # (n_obs,)
    obs_arm: np.ndarray          # (n_obs,) +1 arm_a, -1 arm_b
    obs_k: np.ndarray
    obs_n: np.ndarray
    arm_a: str = "orig"
    arm_b: str = "swap"

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_grid(self) -> int:
        return len(self.grid)


def prepare_data(
    records, grid=DEFAULT_GRID, *, arm_a: str = "orig", arm_b: str = "swap"
) -> SwapData:
    """Index `counts.jsonl` records onto the (unit, grid, arm) design.

    Records with ``n == 0`` are dropped (no parsed continuations). Counts
    that land on the same (unit, grid slot, arm) — which happens when cut
    jitter puts two cuts in one bin — are summed. Records whose ``arm`` is
    neither `arm_a` (+1) nor `arm_b` (-1) are ignored, so a counts file
    holding three arms can be fitted pairwise without rewriting it.
    """
    if arm_a == arm_b:
        raise ValueError("arm_a and arm_b must differ")
    arm_code = {arm_a: 1, arm_b: -1}
    grid = np.asarray(grid, dtype=float)
    unit_index: dict[str, int] = {}
    cell_index: dict[str, int] = {}
    unit_cell: list[int] = []
    unit_is_rollout: list[bool] = []
    agg: dict[tuple[int, int, int], list[int]] = {}

    for rec in records:
        n = int(rec["n"])
        if n <= 0:
            continue
        arm = arm_code.get(rec["arm"])
        if arm is None:
            continue
        unit_id = str(rec["unit_id"])
        cell_id = str(rec["cell_id"])
        if cell_id not in cell_index:
            cell_index[cell_id] = len(cell_index)
        if unit_id not in unit_index:
            unit_index[unit_id] = len(unit_index)
            unit_cell.append(cell_index[cell_id])
            unit_is_rollout.append(not rec.get("is_cell_level", False))
        g = int(np.argmin(np.abs(grid - float(rec.get("frac_sentences", 0.0)))))
        key = (unit_index[unit_id], g, arm)
        slot = agg.setdefault(key, [0, 0])
        slot[0] += int(rec["k"])
        slot[1] += n

    keys = sorted(agg)
    return SwapData(
        unit_ids=[u for u, _ in sorted(unit_index.items(), key=lambda kv: kv[1])],
        cell_ids=[c for c, _ in sorted(cell_index.items(), key=lambda kv: kv[1])],
        grid=grid,
        unit_cell=np.array(unit_cell, dtype=int),
        unit_is_rollout=np.array(unit_is_rollout, dtype=bool),
        obs_unit=np.array([k[0] for k in keys], dtype=int),
        obs_grid=np.array([k[1] for k in keys], dtype=int),
        obs_arm=np.array([k[2] for k in keys], dtype=float),
        obs_k=np.array([agg[k][0] for k in keys], dtype=int),
        obs_n=np.array([agg[k][1] for k in keys], dtype=int),
        arm_a=arm_a,
        arm_b=arm_b,
    )


def replicate_constant_arm(
    records,
    constant_by_cell: dict[str, tuple[int, int]],
    *,
    arm_name: str = "baseline",
    source_arm: str = "neutral",
    grid=DEFAULT_GRID,
) -> list[dict]:
    """Emit `arm_name` records that copy a per-cell constant count onto every
    (unit, grid slot) at which the unit has a `source_arm` observation.

    This is how the estimand ``s_i(t) = p_neutral,i(t) - p_base,q`` is fitted
    with the unchanged two-arm model: the question's cached baseline
    favoured-side count ``(k_base, n_base)`` plays arm B at every t.

    Assumption (stated in the module docstring too): p_base is shared by all
    units of the same (prompt_key, direction) cell — it is the rate of the
    paper's no-bet rollouts landing on that direction's favoured side, and
    no prefix enters it. Each unit gets its *own* replica of the count, so
    per-unit s posteriors carry a p_base uncertainty of the right size
    (n_base ~ 100) but that uncertainty is independent across units rather
    than common to the cell. Population statements about s should be read
    with that in mind; `three_arm.py` reports a shared-draw variant.

    `constant_by_cell` maps ``cell_id -> (k, n)``. Units whose cell has no
    entry are skipped (and reported by the caller). One replica per distinct
    grid slot per unit, so cut jitter that lands two cuts in one bin does not
    double-count the baseline.
    """
    grid = np.asarray(grid, dtype=float)
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for rec in records:
        if rec["arm"] != source_arm or int(rec["n"]) <= 0:
            continue
        cell_id = str(rec["cell_id"])
        if cell_id not in constant_by_cell:
            continue
        g = int(np.argmin(np.abs(grid - float(rec.get("frac_sentences", 0.0)))))
        key = (str(rec["unit_id"]), g)
        if key in seen:
            continue
        seen.add(key)
        k, n = constant_by_cell[cell_id]
        out.append(
            {
                **{
                    f: rec[f]
                    for f in (
                        "unit_id", "is_cell_level", "cell_id", "prompt_key",
                        "direction", "threshold",
                    )
                    if f in rec
                },
                "t": rec.get("t"),
                "arm": arm_name,
                "frac_sentences": float(grid[g]),
                "n_sentences": rec.get("n_sentences", 0),
                "k": int(k),
                "n": int(n),
                "n_unparsed": 0,
                "n_truncated": 0,
                "replicated": True,
            }
        )
    return out


# --- model ----------------------------------------------------------------


def _anchored_walk(pt, name, n_units, n_grid, step_sd, pm):
    """Per-unit random walk over the grid with the g=0 value pinned to 0."""
    if n_grid < 2:
        return pt.zeros((n_units, 1))
    z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=(n_units, n_grid - 1))
    walk = pt.cumsum(z * step_sd, axis=1)
    return pt.concatenate([pt.zeros((n_units, 1)), walk], axis=1)


def _global_walk(pt, name, n_grid, loc, scale, step_sd, pm):
    """Global curve over the grid: free start, Gaussian random-walk steps."""
    start = pm.Normal(f"{name}_0", loc, scale)
    if n_grid < 2:
        return pt.reshape(start, (1,))
    z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=n_grid - 1)
    return pt.concatenate([pt.reshape(start, (1,)), start + pt.cumsum(z * step_sd)])


def build_model(data: SwapData, prior: PriorSetting, *, track_arms: bool = False):
    """Construct the PyMC model. Returns the model object.

    `track_arms=True` additionally records the two arm probabilities
    ``p_a`` / ``p_b`` as deterministics (needed by the three-arm readings:
    where p_neutral sits between p_swap and p_orig, and the shared-baseline
    variant of s). Deterministics do not enter the log-density or the random
    stream, so the draws are unchanged either way; the flag only exists so
    the step-3 idata files keep their exact variable set.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n_u, n_c, n_g = data.n_units, data.n_cells, data.n_grid

    with pm.Model() as model:
        # --- dependence (the estimand) ---
        delta_step = pm.HalfNormal("delta_step_sd", prior.delta_step_scale)
        gamma = _global_walk(
            pt, "gamma", n_g, prior.delta0_loc, prior.delta0_scale, delta_step, pm
        )
        pm.Deterministic("gamma_curve", gamma)

        # Cell deviations are constrained to sum to zero across cells at each
        # grid position. Without that constraint `gamma[g]` and the mean of
        # `nu[:, g]` are only softly separated, which leaves a ridge in the
        # posterior that NUTS reports as divergences.
        delta_cell_sd = pm.HalfNormal("delta_cell_sd", prior.delta_cell_scale)
        nu = delta_cell_sd * pt.transpose(
            pm.ZeroSumNormal("nu_z", sigma=1.0, shape=(n_g, n_c))
        )

        delta_unit_step = pm.HalfNormal(
            "delta_unit_step_sd", prior.delta_unit_step_scale
        )
        w = _anchored_walk(pt, "w", n_u, n_g, delta_unit_step, pm)

        delta = gamma[None, :] + nu[data.unit_cell] + w

        # --- nuisance level ---
        alpha_step = pm.HalfNormal("alpha_step_sd", prior.alpha_step_scale)
        abar = _global_walk(pt, "abar", n_g, 0.0, prior.alpha0_scale, alpha_step, pm)
        alpha_cell_sd = pm.HalfNormal("alpha_cell_sd", prior.alpha_cell_scale)
        ac = alpha_cell_sd * pt.transpose(
            pm.ZeroSumNormal("ac_z", sigma=1.0, shape=(n_g, n_c))
        )
        alpha_unit_step = pm.HalfNormal(
            "alpha_unit_step_sd", prior.alpha_unit_step_scale
        )
        v = _anchored_walk(pt, "v", n_u, n_g, alpha_unit_step, pm)
        alpha = abar[None, :] + ac[data.unit_cell] + v

        # --- observation ---
        eta = (
            alpha[data.obs_unit, data.obs_grid]
            + data.obs_arm * delta[data.obs_unit, data.obs_grid] / 2.0
        )
        pm.Binomial(
            "k",
            n=data.obs_n,
            logit_p=eta,
            observed=data.obs_k,
        )

        # dep on the probability scale, the quantity every claim is made in.
        pm.Deterministic(
            "dep",
            pm.math.sigmoid(alpha + delta / 2.0)
            - pm.math.sigmoid(alpha - delta / 2.0),
        )
        if track_arms:
            pm.Deterministic("p_a", pm.math.sigmoid(alpha + delta / 2.0))
            pm.Deterministic("p_b", pm.math.sigmoid(alpha - delta / 2.0))
    return model


def fit(
    data: SwapData,
    prior: PriorSetting,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 0,
    nuts_sampler: str = "pymc",
    progressbar: bool = False,
    track_arms: bool = False,
):
    """Sample the posterior. Returns an ArviZ InferenceData."""
    import pymc as pm

    with build_model(data, prior, track_arms=track_arms):
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            nuts_sampler=nuts_sampler,
            progressbar=progressbar,
        )


# --- summaries ------------------------------------------------------------


def dep_draws(idata, var: str = "dep") -> np.ndarray:
    """(n_draws, n_units, n_grid) array of posterior `var` values."""
    arr = idata.posterior[var].values
    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def point_summary(
    idata,
    data: SwapData,
    *,
    t: float = 0.2,
    eps: float = DEFAULT_EPS,
    hdi_prob: float = 0.94,
    var: str = "dep",
    draws: np.ndarray | None = None,
) -> list[dict]:
    """Per-unit posterior of the arm difference at one grid position.

    Written for ``s(0.2)`` in step 4A but generic: returns, for every unit,
    the posterior mean, equal-tailed 95% limits, an HDI (`hdi_prob`),
    ``P(x > eps)`` and ``P(|x| < eps)`` of `var` at the grid point nearest
    `t`. `draws` (n_draws, n_units, n_grid) replaces the posterior `var`.
    """
    import arviz as az

    dep = dep_draws(idata, var) if draws is None else np.asarray(draws)
    g = int(np.argmin(np.abs(data.grid - float(t))))
    x = dep[:, :, g]                                   # (draws, units)
    hdi = az.hdi(x[None, :, :], hdi_prob=hdi_prob)    # (units, 2)
    hdi = np.asarray(hdi)
    out = []
    for u, unit_id in enumerate(data.unit_ids):
        xu = x[:, u]
        out.append(
            {
                "unit_id": unit_id,
                "is_rollout": bool(data.unit_is_rollout[u]),
                "cell_id": data.cell_ids[int(data.unit_cell[u])],
                "t": float(data.grid[g]),
                "eps": float(eps),
                "mean": float(xu.mean()),
                "lo95": float(np.percentile(xu, 2.5)),
                "hi95": float(np.percentile(xu, 97.5)),
                f"hdi{int(round(hdi_prob * 100))}": [float(hdi[u, 0]), float(hdi[u, 1])],
                "p_gt_eps": float((xu > eps).mean()),
                "p_abs_lt_eps": float((np.abs(xu) < eps).mean()),
            }
        )
    return out


def plugin_s_draws(
    idata,
    data: SwapData,
    constant_by_cell: dict[str, tuple[int, int]],
    *,
    neutral_var: str,
    seed: int = 0,
) -> np.ndarray:
    """``s = p_neutral - p_base`` assembled post hoc from a fit in which the
    neutral arm is one of the two *varying* arms.

    `neutral_var` is ``"p_b"`` for the (orig, neutral) fit or ``"p_a"`` for
    the (neutral, swap) fit (needs ``track_arms=True``). ``p_base`` is one
    Beta(k+1, n-k+1) draw per cell per posterior draw, shared by every unit
    of the cell, from the cached baseline count ``constant_by_cell[cell] =
    (k, n)``.

    Why this exists: fitting s through a replicated constant arm B makes the
    symmetric (alpha, delta) parameterisation pay two random-walk priors for
    any unit-level movement of p_neutral (alpha and delta must move in
    lockstep to keep alpha - delta/2 fixed), which over-shrinks s — the
    three-arm synthetic test shows 95% intervals covering ~75%. Taking
    p_neutral from a fit where both arms move avoids that, keeps the
    p_base uncertainty at its true n ~ 100 scale, and makes it common to
    the cell. Both versions are reported by `three_arm.py`.
    """
    p_neu = dep_draws(idata, neutral_var)               # (draws, units, grid)
    rng = np.random.default_rng(seed)
    n_draws = p_neu.shape[0]
    p_base_cell = np.empty((n_draws, data.n_cells))
    for c, cell_id in enumerate(data.cell_ids):
        k, n = constant_by_cell[cell_id]
        p_base_cell[:, c] = rng.beta(k + 1, n - k + 1, size=n_draws)
    return p_neu - p_base_cell[:, data.unit_cell][:, :, None]


def population_fractions(
    idata,
    data: SwapData,
    unit_ids,
    *,
    t: float = 0.2,
    eps: float = DEFAULT_EPS,
    prob: float = DEFAULT_PROB,
    var: str = "dep",
    draws: np.ndarray | None = None,
) -> dict:
    """Two population readings of ``x(t) > eps`` / ``|x(t)| < eps`` over a
    subset of units (step 4A: the step-3 early-locked set).

    ``classified_*`` counts units whose posterior probability of the event
    exceeds `prob` (the preregistered classification estimand);
    ``propagated_*`` is the per-draw fraction of units in the event, with its
    posterior mean and equal-tailed 95% interval (the fully propagated
    estimand from the step-3 addendum). Units not present in `data` are
    listed under ``missing``. `draws` (n_draws, n_units, n_grid) replaces
    the posterior `var` when given (used for the shared-baseline s variant).
    """
    index = {u: i for i, u in enumerate(data.unit_ids)}
    present = [u for u in unit_ids if u in index]
    missing = [u for u in unit_ids if u not in index]
    if not present:
        return {"n": 0, "missing": missing}
    rows = [index[u] for u in present]
    dep = dep_draws(idata, var) if draws is None else np.asarray(draws)
    g = int(np.argmin(np.abs(data.grid - float(t))))
    x = dep[:, rows, g]                                # (draws, n_subset)
    above = x > eps
    within = np.abs(x) < eps
    p_above = above.mean(axis=0)
    p_within = within.mean(axis=0)
    frac_above = above.mean(axis=1)
    frac_within = within.mean(axis=1)

    def _prop(f):
        return {
            "mean": float(f.mean()),
            "lo95": float(np.percentile(f, 2.5)),
            "hi95": float(np.percentile(f, 97.5)),
        }

    n = len(present)
    return {
        "n": n,
        "t": float(data.grid[g]),
        "eps": float(eps),
        "prob": float(prob),
        "classified_gt_eps": int((p_above > prob).sum()),
        "classified_gt_eps_frac": float((p_above > prob).mean()),
        "classified_abs_lt_eps": int((p_within > prob).sum()),
        "classified_abs_lt_eps_frac": float((p_within > prob).mean()),
        "classified_undecided": int(((p_above <= prob) & (p_within <= prob)).sum()),
        "propagated_gt_eps": _prop(frac_above),
        "propagated_abs_lt_eps": _prop(frac_within),
        "missing": missing,
    }


def crossing_fraction(dep_curves: np.ndarray, grid: np.ndarray, eps: float) -> np.ndarray:
    """Fraction of CoT at which each curve first falls below `eps`.

    `dep_curves` is (..., n_grid). Linear interpolation between grid points
    gives a continuous crossing position instead of snapping to the grid;
    curves that never fall below `eps` inside the grid return NaN
    (right-censored).

    This is the statistic the commitment claim should rest on. A
    threshold-crossing *test* ("is P(dep < eps) > 0.9 at this grid point")
    throws away most of the information and censors any rollout that commits
    near the end of its CoT; a posterior over the crossing position uses the
    whole curve and reports its own uncertainty.
    """
    below = dep_curves < eps
    out = np.full(dep_curves.shape[:-1], np.nan)
    any_below = below.any(axis=-1)
    first = np.argmax(below, axis=-1)
    # g == 0 means the curve starts below eps: no commitment to localise.
    at_start = any_below & (first == 0)
    out[at_start] = float(grid[0])
    interior = any_below & (first > 0)
    if interior.any():
        idx = first[interior]
        prev = idx - 1
        rows = np.nonzero(interior)
        y1 = dep_curves[(*rows, prev)]
        y0 = dep_curves[(*rows, idx)]
        # y1 >= eps > y0 by construction, so the denominator cannot vanish.
        frac = (y1 - eps) / (y1 - y0)
        out[interior] = grid[prev] + frac * (grid[idx] - grid[prev])
    return out


def commitment_summary(
    idata,
    data: SwapData,
    *,
    eps: float = DEFAULT_EPS,
    prob: float = DEFAULT_PROB,
) -> list[dict]:
    """Per-rollout curve plus the two threshold-crossing summaries.

    `t_first_dependent` is the earliest grid position where
    ``P(dep > eps) > prob`` — the literal quantity in the task spec, and the
    one that answers "does this rollout individually depend on the
    threshold". Because dep(0) is a cell-level identity, it is almost always
    0 for any question whose bias exceeds eps, so on its own it does not
    localise commitment.

    `t_collapse` is the earliest grid position where ``P(dep < eps) > prob``
    — the commitment point in the README's sense ("where dep_i(t)
    collapses"). This is the descriptive curve summary the project wants.

    Both are reported; `None` means the criterion was never met on the grid.
    """
    dep = dep_draws(idata)
    p_above = (dep > eps).mean(axis=0)
    p_below = (dep < eps).mean(axis=0)
    mean = dep.mean(axis=0)
    lo = np.percentile(dep, 2.5, axis=0)
    hi = np.percentile(dep, 97.5, axis=0)
    grid = data.grid

    # Per-draw crossing position: (n_draws, n_units), NaN where censored.
    cross = crossing_fraction(dep, grid, eps)

    out = []
    for u, unit_id in enumerate(data.unit_ids):
        first_dep = next((g for g in range(len(grid)) if p_above[u, g] > prob), None)
        collapse = next((g for g in range(len(grid)) if p_below[u, g] > prob), None)
        cu = cross[:, u]
        observed = cu[~np.isnan(cu)]
        p_cross = float(len(observed) / len(cu))
        if len(observed) >= 2:
            cross_med = float(np.median(observed))
            cross_lo = float(np.percentile(observed, 2.5))
            cross_hi = float(np.percentile(observed, 97.5))
        else:
            cross_med = cross_lo = cross_hi = None
        # "Individually dependent" = credibly above eps anywhere past the
        # empty prefix, i.e. the prompt still moved this rollout's outcome
        # after its own reasoning was in context.
        dependent_after_0 = bool(
            any(p_above[u, g] > prob for g in range(1, len(grid)))
        )
        out.append(
            {
                "unit_id": unit_id,
                "is_rollout": bool(data.unit_is_rollout[u]),
                "cell_id": data.cell_ids[int(data.unit_cell[u])],
                "grid": grid.tolist(),
                "dep_mean": mean[u].tolist(),
                "dep_lo95": lo[u].tolist(),
                "dep_hi95": hi[u].tolist(),
                "p_dep_gt_eps": p_above[u].tolist(),
                "t_first_dependent": None if first_dep is None else float(grid[first_dep]),
                "t_collapse": None if collapse is None else float(grid[collapse]),
                "dependent_after_t0": dependent_after_0,
                # Continuous commitment position: posterior median and 95%
                # interval over the fraction of CoT at which dep falls below
                # eps, plus the posterior probability that it does so at all.
                "commitment_frac": cross_med,
                "commitment_lo95": cross_lo,
                "commitment_hi95": cross_hi,
                "p_commits_within_cot": p_cross,
            }
        )
    return out


def covertness_rate(summaries, *, rollouts_only: bool = True) -> dict:
    """Fraction of rollouts individually credible as threshold-dependent."""
    rows = [s for s in summaries if s["is_rollout"] or not rollouts_only]
    if not rows:
        return {"n": 0, "dependent": 0, "rate": float("nan")}
    dependent = sum(s["dependent_after_t0"] for s in rows)
    return {"n": len(rows), "dependent": dependent, "rate": dependent / len(rows)}


def convergence(idata) -> dict:
    """Worst-case R-hat and bulk ESS over the non-deterministic parameters."""
    import arviz as az

    var_names = [
        v for v in idata.posterior.data_vars
        if v not in ("dep", "gamma_curve", "p_a", "p_b")
    ]
    summ = az.summary(idata, var_names=var_names, kind="diagnostics")
    return {
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "n_divergent": int(idata.sample_stats["diverging"].values.sum()),
    }


def fit_all_priors(
    data: SwapData,
    *,
    informed_dep0: float = 0.62,
    which=("skeptical", "neutral", "informed"),
    **fit_kwargs,
) -> dict:
    """Fit every prior setting and return {name: (idata, summaries)}."""
    settings = prior_settings(informed_dep0)
    out = {}
    for name in which:
        idata = fit(data, settings[name], **fit_kwargs)
        out[name] = (idata, commitment_summary(idata, data))
    return out


def side_by_side(results: dict, data: SwapData) -> str:
    """Human-readable three-prior comparison table."""
    lines = [
        f"{'prior':<11} {'dep(0) mean':>12} {'dep(1) mean':>12} "
        f"{'%dependent':>11} {'median t_collapse':>18} {'max Rhat':>9}"
    ]
    for name, (idata, summaries) in results.items():
        dep = dep_draws(idata)
        rollouts = [i for i, r in enumerate(data.unit_is_rollout) if r]
        dep0 = dep[:, rollouts, 0].mean()
        dep_last = dep[:, rollouts, -1].mean()
        rate = covertness_rate(summaries)["rate"]
        collapses = [
            s["t_collapse"] for s in summaries if s["is_rollout"] and s["t_collapse"] is not None
        ]
        med = np.median(collapses) if collapses else float("nan")
        diag = convergence(idata)
        lines.append(
            f"{name:<11} {dep0:>12.3f} {dep_last:>12.3f} {rate:>10.1%} "
            f"{med:>18.2f} {diag['max_rhat']:>9.3f}"
        )
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------


def load_counts(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument(
        "--prior",
        default="all",
        choices=["all", "skeptical", "neutral", "informed"],
    )
    parser.add_argument("--informed-dep0", type=float, default=0.62)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nuts-sampler", default="pymc")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--out", type=Path, default=None, help="summaries JSON")
    parser.add_argument("--netcdf", type=Path, default=None, help="save idata")
    parser.add_argument("--arm-a", default="orig", help="arm coded +1 (default orig)")
    parser.add_argument("--arm-b", default="swap", help="arm coded -1 (default swap)")
    args = parser.parse_args(argv)

    data = prepare_data(load_counts(args.counts), arm_a=args.arm_a, arm_b=args.arm_b)
    print(
        f"{data.n_units} units ({int(data.unit_is_rollout.sum())} rollouts), "
        f"{data.n_cells} cells, {data.n_grid} grid points, "
        f"{len(data.obs_k)} binomial observations; "
        f"dep = p[{data.arm_a}] - p[{data.arm_b}]"
    )

    which = (
        ("skeptical", "neutral", "informed") if args.prior == "all" else (args.prior,)
    )
    results = fit_all_priors(
        data,
        informed_dep0=args.informed_dep0,
        which=which,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
        nuts_sampler=args.nuts_sampler,
    )
    print()
    print(side_by_side(results, data))

    for name, (idata, _) in results.items():
        print(f"  {name}: {convergence(idata)}")

    if args.out:
        payload = {
            name: {
                "summaries": summaries,
                "covertness": covertness_rate(summaries),
                "convergence": convergence(idata),
            }
            for name, (idata, summaries) in results.items()
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nsummaries -> {args.out}")
    if args.netcdf:
        for name, (idata, _) in results.items():
            path = args.netcdf.with_name(f"{args.netcdf.stem}_{name}.nc")
            idata.to_netcdf(str(path))
            print(f"idata -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
