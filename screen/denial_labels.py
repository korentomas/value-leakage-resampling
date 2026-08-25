#!/usr/bin/env python3
"""Join denial/disclosure category labels onto one model's
main_experiment_accurate rollouts, purely from the paper's already-cached v2
CoT-classification judge -- never calls an API. This is the label side of
the step-3 headline cross-tab (per-rollout counterfactual dependence x
whether the CoT denies/admits the threshold's influence); the dependence
side is handled elsewhere.

--- How this was derived (read from the code, not guessed) ---

Category scheme + judge rubric: donation_bet/cot_categories_common.py lines
13-19 (docstring) and 145-152 (`COT_CATEGORIES`) -- six mutually exclusive
raw labels: UNKNOWN, NO_MENTION, NO_STATEMENT, MENTIONED, NOT_INFLUENCED,
INFLUENCED. The actual rubric text sent to the judge is
`_STATEMENT_PROMPT_TEMPLATE` (cot_categories_common.py:88-113), rendered per
text-source by `statement_prompt()` (cot_categories_common.py:131-142,
imported below) with `{llm_text}` left as a literal placeholder for
`classify_cot` to fill per row. This is the "v2" rubric (MENTIONED added,
UNCLEAR folded into INFLUENCED) -- NOT the older 5-category scheme in
shared/classify_cot.py's own `EXTRACT_STATEMENT_PROMPT_V2`, which is a
different, superseded prompt; donation_bet's own scripts (plot_cot_
categories_v2.py, system_prompts.py) use the cot_categories_common version,
and that's what's actually cached under plot_cot_categories_v2_cache/ (see
below) -- confirmed against the real, downloaded data submodule.

4-bucket display scheme (what "Admits/Mentions/No mention/Denies" means
here): donation_bet/plot_cot_categories_v2.py:248-254 `CATEGORY_TO_SEGMENT`
-- INFLUENCED->Admits, MENTIONED->Mentions, {NO_MENTION,NO_STATEMENT}->No
mention, NOT_INFLUENCED->Denies. UNKNOWN is kept as its own bucket (empty
source text or unparseable judge output), never folded into any of the four
-- reproduced below as CATEGORY_TO_BUCKET.

Cache layout: same content-addressed JsonlJudgeCache machinery as the
estimate-judge cache (shared/judge_jsonl_cache.py), rooted differently.
donation_bet/plot_cot_categories_v2.py:96 sets
`SCRIPT_CACHE_ROOT = data/final_data/plot_cot_categories_v2_cache`; the
per-source cache dir under it is donation_bet/cot_categories_common.py:63-78
`source_statement_cache_dir` (reasoning -> `<root>/<experiment>/
sonnet_extract_statement/<config_hash>/`, via shared/classify_cot.py:180-191
`cot_statement_cache_dir`; answer -> `<root>/<experiment>/
sonnet_extract_statement_from_answer/<config_hash>/`). `config_hash` is
`judge_config_hash(template, judge_config)` (shared/judge_jsonl_cache.py:24-
32), `judge_config` = shared/classify_cot.py:202-208 `_statement_judge_config`
({model: claude-sonnet-4-6, max_tokens: 8000, temperature: 1,
reasoning_effort: disabled}). Confirmed on the real (now fully downloaded,
8.6GB) data submodule: `.../main_experiment_accurate/sonnet_extract_
statement/215c0bd3c82d/` (85,186 rows across all models) and
`.../sonnet_extract_statement_from_answer/<hash>/` (36,553 rows).

Join key: NOT a row index or a rollout id of any kind. shared/classify_cot.py
's `_judge_statement_batch` (~lines 265-293) renders
`statement_prompt_template.format(llm_text=blurred_text)` per row (after
`blur_numbers`, shared/classify_cot.py:46-49, scrubs every digit run to "X"
-- imported here unchanged) and looks the RENDERED PROMPT's hash up in a
content-addressed cache -- exactly the same mechanism as the estimate-judge
cache screen/inventory.py already cross-references. So this script
reconstructs each rollout's rendered classification prompt independently
(blur_numbers(row[source_col]) -> statement_prompt(source_col).format(...))
and looks its hash up directly. This is content-addressed, so it's immune to
row-order/index drift between this script's own `get_main_dfs` call and
whatever order the paper's own classification script happened to run rows
in -- no shared "id" needs to exist for the join to be correct.

Row scope: donation_bet/plot_cot_categories_v2.py:298-306 (`_classify_one`)
restricts classification to `direction in INTERVENTION_DIRECTIONS`
(:216, `["below_good", "above_good"]`) -- baseline rows are NEVER sent to the
statement judge in the paper's own pipeline, so this script marks them
`denial_category=None` with `status="not_classified_upstream_baseline"`
rather than counting them as a coverage gap. Within intervention rows, empty
source text (`classify_cot`, shared/classify_cot.py:371-373 `needs_judge`)
is never judged either and gets `UNKNOWN` locally with no cache lookup at
all -- this script reproduces that: `status="empty_source"` -> bucket
"Unknown", distinct from a genuine cache miss on non-empty text
(`status="cache_miss"`).

Usage:
    python denial_labels.py --model-key qwen3.5-35 --out qwen3.5-35_denial.jsonl
    python denial_labels.py --model-key qwen3.6-35 --source answer --out qwen3.6-35_denial_answer.jsonl
    python denial_labels.py --model-key qwen3.5-35 --data-root /path/to/final_data --out out.jsonl

cache_only semantics: this NEVER calls an API. If main_experiment_accurate's
own rollout cache is incomplete for --model-key, it aborts (same CacheOnlyMiss
guard as repro.py, exit 2) before touching the classification cache at all.
If the rollout cache is fine but classification labels are missing for some
rows, those rows are emitted with denial_category=None and status="cache_miss"
-- never sampled, never judged. If the classification cache directory for
--model-key/--source doesn't exist AT ALL (not just partial), a rough,
clearly-labeled cost estimate for a fresh classification pass is printed
(character-based token approximation, not a tokenizer call) so the on-call
human can decide whether to run one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXTERNAL_REPO = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

import shared.runner as runner  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.get_main_dfs import get_main_dfs  # noqa: E402
from shared.classify_cot import blur_numbers  # noqa: E402
from shared.judge_jsonl_cache import JsonlJudgeCache  # noqa: E402
from donation_bet.cot_categories_common import (  # noqa: E402
    statement_prompt,
    parse_statement_answer,
    source_statement_cache_dir,
    _aggregate_lower_bound_split,
    _pct,
)
from donation_bet.bias_metrics import balanced_bias_bootstrap_ci95  # noqa: E402

DEFAULT_EXPERIMENT = "main_experiment_accurate"
DEFAULT_DATA_ROOT = _EXTERNAL_REPO / "data" / "final_data"
INTERVENTION_DIRECTIONS = ("below_good", "above_good")

# shared/classify_cot.py:202-208 _statement_judge_config() -- reproduced
# rather than imported since it's a zero-arg closure over module constants;
# the values themselves (model/max_tokens/temperature/reasoning_effort) are
# what feeds judge_config_hash, so keeping them byte-identical here is what
# actually matters for hitting the same config_hash directory.
STATEMENT_JUDGE_CONFIG = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 8000,
    "temperature": 1,
    "reasoning_effort": "disabled",
}

# donation_bet/plot_cot_categories_v2.py:248-254 CATEGORY_TO_SEGMENT, renamed
# to the team's Admits/Mentions/No mention/Denies vocabulary. UNKNOWN is its
# own bucket in both schemes (never folded in).
CATEGORY_TO_BUCKET = {
    "INFLUENCED": "Admits",
    "MENTIONED": "Mentions",
    "NO_MENTION": "No mention",
    "NO_STATEMENT": "No mention",
    "NOT_INFLUENCED": "Denies",
    "UNKNOWN": "Unknown",
}
BUCKET_ORDER = ["Admits", "Mentions", "No mention", "Denies", "Unknown"]


def classify_row(text, jcache, template):
    """Returns (raw_category_or_None, bucket_or_None, status)."""
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN", "Unknown", "empty_source"
    blurred = blur_numbers(text)
    rendered = template.format(llm_text=blurred)
    entry = jcache.get(rendered)
    if entry is None:
        return None, None, "cache_miss"
    category = parse_statement_answer(entry.get("answer"))
    return category, CATEGORY_TO_BUCKET.get(category, "Unknown"), "cached"


def _rough_token_estimate(text):
    """Character-count approximation (~4 chars/token), NOT a tokenizer call
    -- this script makes zero network requests, including for cost
    estimation. Good enough to compare candidate row counts, not to bill
    against."""
    return max(1, len(text) // 4) if isinstance(text, str) else 0


def estimate_fresh_pass_cost(rows_needing_judge, template):
    prompt_overhead = _rough_token_estimate(template.replace("{llm_text}", ""))
    total_input = 0
    for text in rows_needing_judge:
        total_input += prompt_overhead + _rough_token_estimate(blur_numbers(text))
    n = len(rows_needing_judge)
    # max_tokens=8000 is the CAP (STATEMENT_JUDGE_CONFIG), not the typical
    # completion length -- thinking is disabled (shared/classify_cot.py:249
    # `thinking={"type": "disabled"}` in the real API call) and the rubric
    # asks for a short "carefully reason, then <answer>" response, so actual
    # completions run far under the cap. 150-600 tokens/row is a guess, not
    # measured -- flagged as such below.
    low_out_est, high_out_est = 150 * n, 600 * n
    return {
        "n_rows": n,
        "model": STATEMENT_JUDGE_CONFIG["model"],
        "approx_input_tokens": total_input,
        "approx_output_tokens_range": (low_out_est, high_out_est),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source", choices=["reasoning", "answer"], default="reasoning",
                        help="which text the denial judge classified (default: reasoning, "
                             "matches CATEGORY_SOURCE_COL in plot_cot_categories_v2.py)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"path to final_data/ (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runner.CACHE_DIR = str(args.data_root / "cache")
    runner.ESTIMATE_JUDGE_CACHE_ROOT = str(args.data_root / "estimate_judge_cache")
    script_cache_root = args.data_root / "plot_cot_categories_v2_cache"

    print(f"model-key:  {args.model_key}")
    print(f"experiment: {args.experiment}")
    print(f"source:     {args.source}")

    try:
        out = get_main_dfs(args.experiment, [args.model_key], cache_only=True)
    except runner.CacheOnlyMiss as e:
        print(f"\nROLLOUT CACHE INCOMPLETE for {args.model_key!r} -- aborting before touching "
              f"the classification cache at all.\n  {e}", file=sys.stderr)
        sys.exit(2)
    df, thresholds, display_name = out[args.model_key]

    cache_dir = source_statement_cache_dir(args.source, args.experiment, script_cache_root)
    cache_dir_exists = Path(cache_dir).is_dir()
    template = statement_prompt(args.source)
    jcache = JsonlJudgeCache(cache_dir, template, STATEMENT_JUDGE_CONFIG)
    print(f"classification cache dir: {cache_dir}  (exists: {cache_dir_exists}, "
          f"{len(jcache.entries)} entries loaded)")

    prompt_keys = THRESHOLD_EXPERIMENTS[args.experiment]["prompts"]

    rows_out = []
    raw_categories = []  # aligned with df row order; feeds the Fig-6-equivalent block below
    bucket_counts = {b: 0 for b in BUCKET_ORDER}
    n_baseline = 0
    n_cache_miss = 0
    miss_texts = []  # for the cost estimate, only populated if cache_dir doesn't exist at all

    for _, row in df.iterrows():
        direction = row["direction"]
        base = {
            "model_key": args.model_key,
            "prompt_key": row["prompt_key"],
            "direction": direction,
            "threshold": row.get("threshold"),
            "estimate": row.get("estimate"),
            "on_good_side": row.get("on_good_side"),
            "reasoning": row.get("reasoning"),
            "answer": row.get("answer"),
            "source": args.source,
        }
        if direction not in INTERVENTION_DIRECTIONS:
            n_baseline += 1
            base.update(denial_category=None, denial_bucket=None,
                        status="not_classified_upstream_baseline")
            rows_out.append(base)
            raw_categories.append("UNKNOWN")  # unused: _lower_bound_split_counts drops non-DIRECTIONS rows anyway
            continue

        text = row[args.source]
        category, bucket, status = classify_row(text, jcache, template)
        if status == "cache_miss":
            n_cache_miss += 1
            if not cache_dir_exists:
                miss_texts.append(text)
        else:
            bucket_counts[bucket] += 1
        base.update(denial_category=category, denial_bucket=bucket, status=status)
        rows_out.append(base)
        # A cache miss has no label; fold it into UNKNOWN for the Fig-6-style
        # aggregate below, same as classify_cot's own "judge response failed
        # to parse" UNKNOWN (shared/classify_cot.py module docstring) -- the
        # aggregate's option-(b) rescale already accounts for an UNKNOWN
        # share, see cot_categories_common.py:227-361 docstring.
        raw_categories.append(category if category is not None else "UNKNOWN")

    with open(args.out, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    n_intervention = len(rows_out) - n_baseline
    n_labeled = n_intervention - n_cache_miss
    print(f"\nWrote {len(rows_out)} rows to {args.out} "
          f"({n_baseline} baseline [not classified upstream], "
          f"{n_intervention} intervention rows).")
    print(f"  {n_labeled}/{n_intervention} intervention rows have a denial label "
          f"({n_cache_miss} cache misses, emitted with denial_category=None).")

    print(f"\nRAW category distribution for {display_name} ({args.experiment}, "
          f"{args.source} source, % of ALL labeled intervention rows -- NOT "
          f"the same statistic as the paper's Fig 6, see below):")
    for bucket in BUCKET_ORDER:
        n = bucket_counts[bucket]
        pct = 100 * n / n_labeled if n_labeled else 0.0
        print(f"  {bucket:12s}  {n:5d}  ({pct:5.1f}%)")

    # --- Fig-6-equivalent: bias-weighted decomposition of the BIASED EXCESS
    # only, via the paper's own donation_bet/cot_categories_common.py
    # `_aggregate_lower_bound_split` (imported, not reimplemented) -- this is
    # the actual statistic Figure 6 plots (bar height = model bias; segments
    # = disclosure category shares of the biased rows only), NOT the raw
    # per-row percentage above. `signed=True` matches SIGNED_STACK's default
    # in donation_bet/plot_cot_categories_v2.py:163.
    cat_df = df.copy()
    cat_df["cot_category"] = raw_categories
    fig6 = _aggregate_lower_bound_split(
        cat_df, prompt_keys, category_col="cot_category",
        label=f"{args.model_key} ({args.source})", signed=True,
    )
    bias, ci_low, ci_high = balanced_bias_bootstrap_ci95(df, prompt_keys=prompt_keys)
    fig6_keys = ["truthful_admission", "mentioned", "unfaithful_omission", "false_denial"]
    fig6_labels = {"truthful_admission": "Admits to bias", "mentioned": "Mentions bias",
                   "unfaithful_omission": "No mention of bias", "false_denial": "Denies bias"}
    print(f"\nFig-6-EQUIVALENT decomposition for {display_name} (bar height = overall bias "
          f"{bias:+.3f}, 95% CI [{bias - ci_low:+.3f}, {bias + ci_high:+.3f}]; segments are "
          f"% of ALL rollouts, summing to the bias -- this is the number to compare against "
          f"the paper's Figure 6, not the raw distribution above):")
    for key in fig6_keys:
        print(f"  {fig6_labels[key]:20s}  {_pct(fig6, key):+6.2f}%")
    print(f"  (n_dir={fig6['n_dir']}, n_unknown={fig6['n_unknown']})")

    if not cache_dir_exists:
        print(f"\nClassification cache dir does not exist at all for "
              f"({args.model_key}, {args.source}) -- estimating a fresh pass "
              f"(rough character-based approximation, NOT a tokenizer call, "
              f"NOT a cost quote):")
        est = estimate_fresh_pass_cost(miss_texts, template)
        lo, hi = est["approx_output_tokens_range"]
        print(f"  rows needing judging:      {est['n_rows']}")
        print(f"  judge model:               {est['model']}")
        print(f"  approx. input tokens:      {est['approx_input_tokens']:,}")
        print(f"  approx. output tokens:     {lo:,} - {hi:,}  (150-600/row guess, not measured; "
              f"max_tokens cap is 8000/row but thinking is disabled so real completions run far "
              f"under that)")
        print("  Look up current claude-sonnet-4-6 per-token pricing before deciding "
              "whether to run this.")


if __name__ == "__main__":
    main()
