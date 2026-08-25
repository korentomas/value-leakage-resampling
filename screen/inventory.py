#!/usr/bin/env python3
"""Read-only inventory of the value_leakage_data cache tree
(external/value_leakage/data/final_data once the submodule download lands).

Reports, per model and question, what's cached on both sides of the
pipeline: the model-output rollout cache and the estimate-judge cache. Never
writes anything; tolerant of a partially-downloaded tree (missing dirs,
truncated files).

--- Cache layout, derived from the paper's own code (not guessed) ---

Model-output cache:
    <CACHE_DIR>/<model_name>/<prompt_key>/<direction>_<hash>.jsonl
    (shared/runner.py:87-88 `_cache_path`). First line is a meta JSON dict
    (written by shared/runner.py:107-113 `_write_cache`); baseline files get
    meta keys hash/model_name/prompt_key/direction/n (written from
    `_get_baseline`, shared/runner.py:974-980), direction files get
    hash/model_name/prompt_key/direction/n_per_threshold/thresholds (written
    from `_run_directions_for_prompt`, shared/runner.py:1148-1155). Every
    following line is one rollout dict (reasoning, answer, prompt[,
    threshold for direction files]) -- confirmed by the reader,
    shared/runner.py:91-104 `_read_cache`.

Estimate-judge cache:
    <ESTIMATE_JUDGE_CACHE_ROOT>/<experiment_name>/<config_hash>/<shard>.jsonl
    (shared/runner.py:23-29 `estimate_judge_cache_dir`, called from
    `batch_extract_estimates`). `config_hash` is
    shared/judge_jsonl_cache.py:24-32 `judge_config_hash`, a hash of
    {model, max_tokens, temperature, reasoning_effort, prompt=<judge
    TEMPLATE text, not the rendered per-row prompt>}. Since all 7 "number"
    questions share one judge template (`_NUMBER_JUDGE_PROMPT`,
    shared/prompts/thresholds.py) and only whale/windowdays use the "days"
    template, there are at most 2 config_hash buckets per judge-model
    config actually used historically. `<shard>` is the first hex char of
    shared/judge_jsonl_cache.py:37,66-68 `prompt_hash`/`_shard_prefix`
    applied to the *rendered* judge prompt. Each row is
    `{"answer": ..., "prompt_hash": ...}` -- prompt text and prompt_key are
    deliberately NOT stored (module docstring, shared/judge_jsonl_cache.py
    lines 1-6), so a judge-cache row can only be attributed to a specific
    (model, prompt_key, direction, row) by recomputing
    prompt_hash(judge_template.format(llm_text=<that row's answer>)) and
    checking membership -- which is exactly what the "judge coverage"
    section below does, restricted to main_experiment_accurate's 9
    questions.

DATA_ROOT convention (donation_bet/get_data.py:21,48-49; same pattern in
donation_bet/plot_biases.py:13-18): <repo>/data/final_data/cache and
<repo>/data/final_data/estimate_judge_cache.

Usage:
    python inventory.py                       # default --data-root
    python inventory.py --data-root /path/to/data/final_data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_EXTERNAL_REPO = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

from shared.models import MODELS  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.prompts import THRESHOLD_PROMPTS  # noqa: E402
from shared.judge_jsonl_cache import prompt_hash as _judge_prompt_hash  # noqa: E402

DEFAULT_DATA_ROOT = _EXTERNAL_REPO / "data" / "final_data"
FOCUS_EXPERIMENT = "main_experiment_accurate"
# Beyond qwen3.5-35/qwen3.6-35, these open-weight model keys exist in
# shared/models.py and might have cached main_experiment_accurate rollouts
# worth screening for free (see ../FINDINGS.md "Model keys" note).
EXTRA_MODELS_OF_INTEREST = ["kimi-k2.6", "kimi-k2.5", "deepseek-v3.1",
                            "gpt-oss-120b", "nemotron3-120b", "qwen3.5-397"]


# --- Readers, tolerant of partial/truncated files ---

def _read_model_output_file(path):
    """(meta_dict_or_None, [row_dict, ...]). First line is meta (may be
    missing/corrupt on a truncated download); every following non-blank
    line is one rollout row. Malformed lines are skipped, not fatal."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return None, []
    if not lines:
        return None, []
    meta = None
    data_lines = lines
    try:
        meta = json.loads(lines[0])
        data_lines = lines[1:]
    except json.JSONDecodeError:
        pass  # meta line missing/corrupt -- still try to salvage row lines below
    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return meta, rows


def _read_judge_shard(path):
    """Judge cache shards have no meta line -- every line is a row
    (shared/judge_jsonl_cache.py `load_jsonl_cache`/`append_jsonl_rows`)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --- Scanners ---

def scan_model_output_cache(cache_root: Path):
    """{model_name: {prompt_key: {direction: {"rows": [...], "meta": dict|None, "path": Path}}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    if not cache_root.is_dir():
        return out
    for model_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        for pk_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for f in sorted(pk_dir.glob("*.jsonl")):
                meta, rows = _read_model_output_file(f)
                # Filename is "<direction>_<hash>.jsonl" (_cache_path); prefer
                # the meta's own "direction" field when present since it's
                # authoritative, fall back to parsing the filename.
                direction = (meta or {}).get("direction")
                if not direction:
                    stem = f.stem
                    direction = stem.rsplit("_", 1)[0] if "_" in stem else stem
                out[model_dir.name][pk_dir.name][direction] = {
                    "rows": rows, "meta": meta, "path": f,
                }
    return out


def scan_judge_cache(judge_root: Path):
    """{experiment_name: {config_hash: {prompt_hash: row}}}"""
    out = defaultdict(dict)
    if not judge_root.is_dir():
        return out
    for exp_dir in sorted(p for p in judge_root.iterdir() if p.is_dir()):
        for config_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            entries = {}
            for shard in sorted(config_dir.glob("*.jsonl")):
                for row in _read_judge_shard(shard):
                    h = row.get("prompt_hash")
                    if isinstance(h, str):
                        entries[h] = row
            out[exp_dir.name][config_dir.name] = entries
    return out


# --- Reporting ---

def report_model_output_cache(model_cache):
    print("=" * 78)
    print(f"MODEL-OUTPUT CACHE  ({len(model_cache)} model dir(s) found)")
    print("=" * 78)
    if not model_cache:
        print("  (cache/ missing or empty -- submodule not downloaded yet, or wrong --data-root)")
        return

    focus_pks = THRESHOLD_EXPERIMENTS[FOCUS_EXPERIMENT]["prompts"]

    for model_name in sorted(model_cache):
        pk_map = model_cache[model_name]
        flag = ""
        if model_name not in ("qwen3.5-35", "qwen3.6-35") and model_name in EXTRA_MODELS_OF_INTEREST:
            flag = "  <-- EXTRA (beyond qwen3.5-35/qwen3.6-35), check gate relevance"
        elif model_name not in MODELS:
            flag = "  <-- not in current shared/models.py MODELS (stale/renamed key?)"
        print(f"\n[{model_name}]{flag}  -- {len(pk_map)} prompt_key dir(s) on disk")

        n_full_questions = 0
        for pk in focus_pks:
            cell = pk_map.get(pk)
            if cell is None:
                print(f"    {pk:28s}  NOT CACHED")
                continue
            b = cell.get("baseline")
            bg = cell.get("below_good")
            ag = cell.get("above_good")
            b_n = len(b["rows"]) if b else 0
            bg_n = len(bg["rows"]) if bg else 0
            ag_n = len(ag["rows"]) if ag else 0
            complete = bool(b) and bool(bg) and bool(ag)
            n_full_questions += int(complete)
            status = "OK" if complete else "PARTIAL"
            print(f"    {pk:28s}  baseline={b_n:<4d} below_good={bg_n:<4d} above_good={ag_n:<4d}  [{status}]")

        print(f"  -> {FOCUS_EXPERIMENT}: {n_full_questions}/{len(focus_pks)} questions have "
              f"baseline+both directions cached")

        # Other experiments this model has ANY cache for (prompt_key sets
        # beyond main_experiment_accurate) -- cheap completeness check against
        # every defined experiment, since prompt_key dirs are shared cache
        # keyspace across experiments that happen to reuse the same prompt_key.
        other_hits = []
        for exp_name, exp_cfg in THRESHOLD_EXPERIMENTS.items():
            if exp_name == FOCUS_EXPERIMENT:
                continue
            exp_pks = exp_cfg["prompts"]
            present = sum(1 for pk in exp_pks if pk in pk_map)
            if present:
                other_hits.append((exp_name, present, len(exp_pks)))
        if other_hits:
            other_hits.sort(key=lambda t: -t[1] / t[2])
            print("  Also has cache overlapping these other experiments:")
            for exp_name, present, total in other_hits[:8]:
                print(f"    {exp_name}: {present}/{total} prompt_keys present")
            if len(other_hits) > 8:
                print(f"    ... and {len(other_hits) - 8} more")


def report_judge_cache(judge_cache):
    print("\n" + "=" * 78)
    print(f"ESTIMATE-JUDGE CACHE  ({len(judge_cache)} experiment dir(s) found)")
    print("=" * 78)
    if not judge_cache:
        print("  (estimate_judge_cache/ missing or empty)")
        return
    for exp_name in sorted(judge_cache):
        print(f"\n[{exp_name}]")
        for config_hash, entries in sorted(judge_cache[exp_name].items()):
            print(f"    config_hash={config_hash}: {len(entries)} judged rows")


def report_judge_coverage(model_cache, judge_cache):
    """Cross-reference: for each model x main_experiment_accurate question x
    direction, how many of the on-disk rollout rows have a matching judged
    estimate anywhere in that experiment's judge cache (tried across every
    config_hash bucket found, since we don't know a priori which judge-model
    config produced the cache -- see module docstring)."""
    print("\n" + "=" * 78)
    print(f"JUDGE COVERAGE for {FOCUS_EXPERIMENT}  "
          "(cross-referenced by recomputing each row's judge-prompt hash)")
    print("=" * 78)

    exp_judge_buckets = judge_cache.get(FOCUS_EXPERIMENT, {})
    if not exp_judge_buckets:
        print(f"  No judge cache dir for {FOCUS_EXPERIMENT!r} -- nothing to cross-reference.")
        return
    all_judged_hashes = set()
    for entries in exp_judge_buckets.values():
        all_judged_hashes.update(entries.keys())

    focus_pks = THRESHOLD_EXPERIMENTS[FOCUS_EXPERIMENT]["prompts"]
    for model_name in sorted(model_cache):
        pk_map = model_cache[model_name]
        if not any(pk in pk_map for pk in focus_pks):
            continue
        print(f"\n[{model_name}]")
        for pk in focus_pks:
            cell = pk_map.get(pk)
            if cell is None:
                continue
            if pk not in THRESHOLD_PROMPTS:
                print(f"    {pk:28s}  SKIP (prompt_key not in current THRESHOLD_PROMPTS -- stale?)")
                continue
            judge_template = THRESHOLD_PROMPTS[pk]["judge_prompt"]
            for direction in ("baseline", "below_good", "above_good"):
                sub = cell.get(direction)
                if not sub:
                    continue
                rows = sub["rows"]
                n_judged = 0
                for row in rows:
                    answer = row.get("answer", "")
                    rendered = judge_template.format(llm_text=answer)
                    if _judge_prompt_hash(rendered) in all_judged_hashes:
                        n_judged += 1
                print(f"    {pk:28s} {direction:10s}  {n_judged}/{len(rows)} rows have a judged estimate")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"path to final_data/ (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--skip-judge-coverage", action="store_true",
                        help="skip the hash-recompute cross-reference (faster on a huge cache; "
                             "row-count reports still run)")
    args = parser.parse_args()

    data_root = args.data_root
    print(f"data-root: {data_root}  (exists: {data_root.is_dir()})")

    cache_root = data_root / "cache"
    judge_root = data_root / "estimate_judge_cache"

    model_cache = scan_model_output_cache(cache_root)
    judge_cache = scan_judge_cache(judge_root)

    report_model_output_cache(model_cache)
    report_judge_cache(judge_cache)
    if not args.skip_judge_coverage:
        report_judge_coverage(model_cache, judge_cache)

    print("\n" + "=" * 78)
    ready = [m for m in ("qwen3.5-35", "qwen3.6-35") if m in model_cache]
    print(f"Models with any main_experiment_accurate cache present: {ready or 'none'}")
    print("Run repro.py per model to check cache_only completeness + reproduction against the "
          "published 0.62/0.27 numbers.")


if __name__ == "__main__":
    main()
