#!/usr/bin/env python3
"""Step-1 gate: reproduce the paper's OWN published Donation Bet bias numbers
(0.62 for qwen3.5-35, 0.27 for qwen3.6-35 -- Fig. 4, Betley et al. arXiv
2607.14345) purely from their cached rollouts + judge outputs. This is a
much tighter check than screen/bias.py's ±0.15 sampling gate: if this
mismatches by more than a couple hundredths, either the cache tree is
incomplete/wrong, or something about how bias is being computed here has
drifted from their own code -- since this uses their exact
get_main_dfs/balanced_bias_score/balanced_bias_bootstrap_ci95, drift
shouldn't be possible, so a big miss here means a cache/data problem, not a
methodology problem.

cache_only=True (the same flag donation_bet/get_data.py and
donation_bet/plot_biases.py use) makes shared.get_main_dfs's model-output
sampling AND judge-extraction both raise `CacheOnlyMiss`
(defined shared/runner.py:47; raised at shared/runner.py:966 in
`_get_baseline` on a baseline-rollout miss, :1130 in
`_run_directions_for_prompt` on a direction-rollout miss, and :1048 in
`batch_extract_estimates` on a judge-cache miss) instead of ever calling a
model or a judge API. This script never disables that flag, so it can never
spend API money or sample -- a CacheOnlyMiss is caught and reported as
"cache incomplete", not as a crash.

Usage:
    python repro.py --model-key qwen3.5-35
    python repro.py --model-key qwen3.6-35 --data-root /path/to/final_data
    python repro.py --model-key qwen3.5-35 --tolerance 0.03

Exit codes:
    0  reproduced within tolerance
    1  reproduced FROM cache, but the number doesn't match published within tolerance
    2  cache incomplete for this model -- CacheOnlyMiss, nothing was sampled/judged
    3  bad arguments (unknown --model-key, etc.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXTERNAL_REPO = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

import shared.runner as runner  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.get_main_dfs import get_main_dfs  # noqa: E402
from donation_bet.bias_metrics import (  # noqa: E402
    balanced_bias_score,
    balanced_bias_bootstrap_ci95,
)

DEFAULT_EXPERIMENT = "main_experiment_accurate"
DEFAULT_DATA_ROOT = _EXTERNAL_REPO / "data" / "final_data"
DEFAULT_TOLERANCE = 0.02

# Published Fig. 4 numbers (Betley et al., arXiv 2607.14345).
PUBLISHED_BIAS = {
    "qwen3.5-35": 0.62,
    "qwen3.6-35": 0.27,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True, choices=sorted(PUBLISHED_BIAS),
                        help="model to reproduce (only ones with a published number to check against)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"path to final_data/ (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="max |measured - published| to count as reproduced (default 0.02 -- "
                             "this checks reproduction of THEIR cache, not our own sampling gate)")
    args = parser.parse_args()

    if args.model_key not in PUBLISHED_BIAS:
        print(f"No published number for {args.model_key!r}; nothing to reproduce against.", file=sys.stderr)
        sys.exit(3)

    # Same redirect donation_bet/get_data.py:48-49 and donation_bet/
    # plot_biases.py:17-18 use to point shared.runner at the data submodule's
    # cache instead of the (committed, empty-here) shared/cache tree. Both
    # knobs are read at call time by get_main_dfs/run_thresholds_experiment,
    # so reassigning here before calling get_main_dfs is sufficient -- no
    # monkeypatching needed.
    runner.CACHE_DIR = str(args.data_root / "cache")
    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(args.data_root / "estimate_judge_cache")

    print(f"model-key:  {args.model_key}")
    print(f"experiment: {args.experiment}")
    print(f"data-root:  {args.data_root}")
    print(f"  model cache: {runner.CACHE_DIR}")
    print(f"  judge cache: {runner.estimate_judge_cache_dir(args.experiment)}")

    try:
        out = get_main_dfs(args.experiment, [args.model_key], cache_only=True)
    except runner.CacheOnlyMiss as e:
        print(f"\nCACHE INCOMPLETE -- aborting without sampling or judging anything.\n  {e}",
              file=sys.stderr)
        sys.exit(2)

    df, thresholds, display_name = out[args.model_key]
    prompt_keys = THRESHOLD_EXPERIMENTS[args.experiment]["prompts"]

    print(f"\nLoaded {len(df)} direction rows for {display_name} entirely from cache.")

    overall_bias, ci_low, ci_high = balanced_bias_bootstrap_ci95(df, prompt_keys=prompt_keys)
    print(f"\nOverall balanced_bias_score ({args.experiment}, {len(prompt_keys)} questions):")
    print(f"  measured  = {overall_bias:+.4f}  "
          f"(95% CI: [{overall_bias - ci_low:+.4f}, {overall_bias + ci_high:+.4f}])")

    print("\n  per-question:")
    for pk in prompt_keys:
        pk_df = df[df["prompt_key"] == pk]
        if pk_df.empty:
            print(f"    {pk}: NO ROWS")
            continue
        print(f"    {pk}: {balanced_bias_score(pk_df):+.4f}  (n={len(pk_df)})")

    published = PUBLISHED_BIAS[args.model_key]
    diff = overall_bias - published
    passed = abs(diff) <= args.tolerance
    status = "REPRODUCED" if passed else "MISMATCH"
    print(f"\nPublished: {published:+.4f}   Measured: {overall_bias:+.4f}   "
          f"Diff: {diff:+.4f}   Tolerance: +/-{args.tolerance}")
    print(f"  -> {status}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
