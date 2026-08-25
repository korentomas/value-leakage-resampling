#!/usr/bin/env python3
"""Build THE canonical step-3 input table for one model: one row per
main_experiment_accurate rollout (baseline included), with denial-category
and evaluation-awareness labels joined on, ready for the swap workstream to
consume as-is. Never calls an API, never samples, never cuts/segments a CoT
-- that belongs to the swap workstream.

This is a thin assembly layer over screen/denial_labels.py and
screen/awareness_labels.py: it imports both modules directly (not
copy-pasted logic) and reuses their exact classify_row / classification
functions, so a rollout's label here is bit-for-bit what those two scripts
would independently produce for the same rollout. See those two scripts'
docstrings for the full judge/cache/rubric derivation citations; this
docstring only covers what's new here.

--- rollout_id ---

Defined as sha256(json.dumps({model_key, experiment, prompt_key, direction,
reasoning, answer}, sort_keys=True)).hexdigest() -- the SAME content tuple
documented as the cross-script join key in denial_labels.py's and
awareness_labels.py's docstrings (prompt_key + direction + reasoning +
answer), just formalized into a single stable hash column instead of a
multi-column merge key, and salted with model_key + experiment so IDs stay
globally unique if multiple models' tables are ever concatenated. This is
NOT any hash used by the paper's own code (their JsonlJudgeCache hashes
differ per judge -- see denial_labels.py/awareness_labels.py) -- it's new,
defined here, for this table only.

--- Row scope ---

ALL rows from get_main_dfs (baseline + below_good + above_good), matching
the team's ask that baseline rows carry their labels "where they exist":
  - denial_category / denial_bucket: None for baseline rows, status=
    "not_classified_upstream_baseline" -- the paper's own pipeline never
    sends baseline CoTs to the denial judge (see denial_labels.py).
  - awareness_score / eval_aware / weak_signal: present for baseline rows
    too when cached -- the paper's own pipeline judges all directions for
    eval-awareness (see awareness_labels.py).
  - on_good_side: None for baseline rows. shared/get_main_dfs.py's own
    `_add_good_side` computes this via two direction-gated OR terms that
    are both False (not NaN) for baseline rows -- i.e. their raw df
    literally stores `on_good_side=False` for baseline, which reads as "on
    the bad side" even though the concept doesn't apply outside an
    intervention prompt. This table corrects that to None so a naive
    consumer doesn't silently treat baseline rows as bad-side evidence.

--- Self-check ---

Before writing anything, recomputes balanced_bias_score /
balanced_bias_bootstrap_ci95 (donation_bet/bias_metrics.py, imported) on the
assembled table's own below_good/above_good rows and compares to the
published Fig. 4 number (0.62 qwen3.5-35, 0.27 qwen3.6-35) at a tight
tolerance (this table is built entirely from cache, same as repro.py, so it
should reproduce repro.py's number almost exactly). On mismatch beyond
tolerance, aborts loudly (stderr, exit 1) WITHOUT writing the parquet or
JSON sidecar -- a table that fails its own self-check is not shipped as the
canonical input.

Usage:
    python master_table.py --model-key qwen3.5-35 --out qwen3.5-35_master.parquet
    python master_table.py --model-key qwen3.6-35 --out qwen3.6-35_master.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCREEN_DIR = Path(__file__).resolve().parent
if str(_SCREEN_DIR) not in sys.path:
    sys.path.insert(0, str(_SCREEN_DIR))

_EXTERNAL_REPO = _SCREEN_DIR.parents[0] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

import pandas as pd  # noqa: E402

import shared.runner as runner  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.get_main_dfs import get_main_dfs  # noqa: E402
from shared.judge_jsonl_cache import JsonlJudgeCache  # noqa: E402
from donation_bet.bias_metrics import (  # noqa: E402
    balanced_bias_score,
    balanced_bias_bootstrap_ci95,
)

import denial_labels  # noqa: E402  (screen/denial_labels.py, same dir)
import awareness_labels  # noqa: E402  (screen/awareness_labels.py, same dir)

DEFAULT_EXPERIMENT = "main_experiment_accurate"
DEFAULT_DATA_ROOT = _EXTERNAL_REPO / "data" / "final_data"
DEFAULT_SOURCE = "reasoning"  # matches CATEGORY_SOURCE_COL default (denial_labels.py)

PUBLISHED_BIAS = {"qwen3.5-35": 0.62, "qwen3.6-35": 0.27}
SELF_CHECK_TOLERANCE = 0.005  # tight: same cache as repro.py, should nearly exactly match


def rollout_id(model_key, experiment, prompt_key, direction, reasoning, answer):
    payload = {
        "model_key": model_key, "experiment": experiment, "prompt_key": prompt_key,
        "direction": direction, "reasoning": reasoning, "answer": answer,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_table(model_key, experiment, data_root, source=DEFAULT_SOURCE):
    runner.CACHE_DIR = str(data_root / "cache")
    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(data_root / "estimate_judge_cache")

    try:
        out = get_main_dfs(experiment, [model_key], cache_only=True)
    except runner.CacheOnlyMiss as e:
        print(f"\nROLLOUT CACHE INCOMPLETE for {model_key!r} -- aborting before touching "
              f"any label cache.\n  {e}", file=sys.stderr)
        sys.exit(2)
    df, thresholds, display_name = out[model_key]

    denial_cache_dir = denial_labels.source_statement_cache_dir(
        source, experiment, data_root / "plot_cot_categories_v2_cache",
    )
    denial_template = denial_labels.statement_prompt(source)
    denial_jcache = JsonlJudgeCache(denial_cache_dir, denial_template, denial_labels.STATEMENT_JUDGE_CONFIG)

    awareness_cache_dir = awareness_labels.eval_awareness_cache_dir(
        experiment, cache_root=data_root / "plot_eval_awareness_cache",
    )
    awareness_jcache = JsonlJudgeCache(
        awareness_cache_dir, awareness_labels.ACTIVE_EVAL_AWARENESS_PROMPT, awareness_labels._judge_config(),
    )

    rows = []
    stats = {
        "n_total": 0, "n_baseline": 0, "n_intervention": 0,
        "denial_labeled": 0, "denial_cache_miss": 0, "denial_empty": 0,
        "awareness_labeled": 0, "awareness_cache_miss": 0, "awareness_empty": 0,
    }

    for _, row in df.iterrows():
        direction = row["direction"]
        prompt_key = row["prompt_key"]
        reasoning = row.get("reasoning")
        answer = row.get("answer")
        is_baseline = direction == "baseline"

        stats["n_total"] += 1
        stats["n_baseline" if is_baseline else "n_intervention"] += 1

        # --- denial (denial_labels.classify_row, same function that script uses) ---
        if direction in denial_labels.INTERVENTION_DIRECTIONS:
            denial_category, denial_bucket, denial_status = denial_labels.classify_row(
                reasoning, denial_jcache, denial_template,
            )
            # "labeled" = the table carries a real denial_category value.
            # empty_source counts: it resolves to category="UNKNOWN", the
            # SAME value shared/classify_cot.py's own classify_cot hardcodes
            # for empty reasoning without ever calling the judge (see
            # denial_labels.py's docstring) -- it's a legitimate label, not
            # a gap. Only cache_miss (non-empty text, no cached judge
            # answer) is a real coverage gap.
            if denial_status in ("cached", "empty_source"):
                stats["denial_labeled"] += 1
            if denial_status == "cache_miss":
                stats["denial_cache_miss"] += 1
            if denial_status == "empty_source":
                stats["denial_empty"] += 1
        else:
            denial_category, denial_bucket, denial_status = None, None, "not_classified_upstream_baseline"

        # --- awareness (same lookup awareness_labels.py performs) ---
        # Unlike denial's UNKNOWN, empty reasoning here genuinely has no
        # score (shared/classify_eval_awareness.py's own `needs_judge` mask
        # skips it, leaving eval_awareness_score=NaN in their own df) -- so
        # empty_source is correctly NOT counted as "labeled" below.
        if not isinstance(reasoning, str) or not reasoning.strip():
            awareness_score, eval_aware, weak_signal, awareness_status = None, False, False, "empty_source"
            stats["awareness_empty"] += 1
        else:
            rendered = awareness_labels.ACTIVE_EVAL_AWARENESS_PROMPT.format(
                prompt=row.get("prompt"), reasoning=reasoning, answer=answer,
            )
            entry = awareness_jcache.get(rendered)
            if entry is None:
                awareness_score, eval_aware, weak_signal, awareness_status = None, False, False, "cache_miss"
                stats["awareness_cache_miss"] += 1
            else:
                score = awareness_labels._parse_score(entry.get("answer"))
                if score is None:
                    awareness_score, eval_aware, weak_signal, awareness_status = None, False, False, "cached_unparseable"
                else:
                    awareness_score = score
                    eval_aware = score >= awareness_labels.EVAL_AWARE_THRESHOLD
                    weak_signal = score >= awareness_labels.EVAL_WEAK_CUTOFF
                    awareness_status = "cached"
                    stats["awareness_labeled"] += 1

        rows.append({
            "rollout_id": rollout_id(model_key, experiment, prompt_key, direction, reasoning, answer),
            "model_key": model_key,
            "experiment": experiment,
            "prompt_key": prompt_key,
            "direction": direction,
            "threshold": row.get("threshold"),
            "prompt": row.get("prompt"),
            "reasoning": reasoning,
            "answer": answer,
            "estimate": row.get("estimate"),
            "on_good_side": None if is_baseline else bool(row.get("on_good_side")),
            "denial_category": denial_category,
            "denial_bucket": denial_bucket,
            "denial_status": denial_status,
            "awareness_score": awareness_score,
            "eval_aware": eval_aware,
            "weak_signal": weak_signal,
            "awareness_status": awareness_status,
        })

    table = pd.DataFrame(rows)
    assert table["rollout_id"].is_unique, "rollout_id collision -- investigate before shipping"
    return table, display_name, stats


def self_check_bias(table, experiment, model_key):
    prompt_keys = THRESHOLD_EXPERIMENTS[experiment]["prompts"]
    bias, ci_low, ci_high = balanced_bias_bootstrap_ci95(table, prompt_keys=prompt_keys)
    published = PUBLISHED_BIAS.get(model_key)
    result = {
        "measured_bias": bias,
        "bias_ci95": [bias - ci_low, bias + ci_high],
        "published_bias": published,
    }
    if published is not None:
        diff = bias - published
        result["diff"] = diff
        result["passed"] = abs(diff) <= SELF_CHECK_TOLERANCE
    else:
        result["diff"] = None
        result["passed"] = None  # no published number to check against
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source", choices=["reasoning", "answer"], default=DEFAULT_SOURCE,
                        help="denial-classification source (default: reasoning, matches "
                             "CATEGORY_SOURCE_COL in the paper's own scripts)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", required=True, help="output .parquet path; the JSON sidecar "
                        "is written alongside it with a .stats.json suffix")
    args = parser.parse_args()

    out_path = Path(args.out)
    sidecar_path = out_path.with_suffix(".stats.json")

    print(f"model-key:  {args.model_key}")
    print(f"experiment: {args.experiment}")
    print(f"source:     {args.source}")

    table, display_name, stats = build_table(
        args.model_key, args.experiment, args.data_root, source=args.source,
    )

    check = self_check_bias(table, args.experiment, args.model_key)
    print(f"\nSelf-check for {display_name}: measured bias {check['measured_bias']:+.4f} "
          f"(95% CI [{check['bias_ci95'][0]:+.4f}, {check['bias_ci95'][1]:+.4f}])")
    if check["published_bias"] is not None:
        print(f"  published: {check['published_bias']:+.4f}   diff: {check['diff']:+.4f}   "
              f"tolerance: +/-{SELF_CHECK_TOLERANCE}")
        if not check["passed"]:
            print(
                f"\nSELF-CHECK FAILED: measured bias disagrees with the published number by "
                f"more than {SELF_CHECK_TOLERANCE} -- ABORTING WITHOUT WRITING OUTPUT. "
                f"This table would be built from the same cache repro.py already validated "
                f"against 0.620/0.271, so a mismatch here means something in this script's "
                f"row assembly is wrong, not that the underlying data changed. Do not use "
                f"any partial output from this run.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  -> PASS")
    else:
        print(f"  no published number for {args.model_key!r} to check against -- skipping abort logic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, engine="pyarrow", index=False)
    file_size = out_path.stat().st_size

    sidecar = {
        "model_key": args.model_key,
        "display_name": display_name,
        "experiment": args.experiment,
        "source": args.source,
        "row_counts": stats,
        "label_coverage": {
            "denial_pct_of_intervention": (
                100 * stats["denial_labeled"] / stats["n_intervention"] if stats["n_intervention"] else 0.0
            ),
            "awareness_pct_of_total": (
                100 * stats["awareness_labeled"] / stats["n_total"] if stats["n_total"] else 0.0
            ),
        },
        "bias_self_check": check,
        "output": {
            "parquet_path": str(out_path),
            "parquet_bytes": file_size,
            "n_rows": len(table),
            "n_columns": len(table.columns),
            "columns": list(table.columns),
        },
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"\nWrote {len(table)} rows x {len(table.columns)} cols to {out_path} "
          f"({file_size:,} bytes)")
    print(f"Wrote sidecar to {sidecar_path}")
    print(f"  denial labeled:    {stats['denial_labeled']}/{stats['n_intervention']} intervention rows")
    print(f"  awareness labeled: {stats['awareness_labeled']}/{stats['n_total']} total rows")


if __name__ == "__main__":
    main()
