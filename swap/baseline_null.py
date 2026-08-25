"""Step 4D: null distribution for s — prefixes cut from *no-bet* rollouts.

s(t) = P(favoured | neutral prefix) − p_base has no null in steps 3/4A: a
prefix cut from a bet rollout at t = 0.2 is compared against the paper's
baseline rate, but a prefix cut from a *baseline* rollout at the same t might
also have "picked a side" already. This script builds that null arm:

  * 250 of the paper's cached Qwen3.5 baseline (no-bet) rollouts, uniform
    over the 9 questions, drawn with seed 0 from the master parquet;
  * each cut with exactly the step-3 rule (`plan_cuts`, n_cuts=6, seed 0,
    jitter) and only the second cut (t ≈ 0.2) kept;
  * continued ×25 under the same baseline prompt (arm ``neutral_null``),
    temperature 1, max_tokens = 16000 − prefix tokens.

Scoring: ``direction = "baseline"`` has no favoured side, so `driver.
favored_side` scores it as ``estimate > threshold`` with the question's
(unique, median) threshold; ``s_null = k/n − 0.5`` per prefix, reported as
|s_null| (both sides symmetric).

Usage (from swap/):
    ../external/value_leakage/.venv/bin/python baseline_null.py build
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from cuts import plan_cuts
from driver import _continuation_body, _req_id, _threshold_prompts, _write_jsonl
from template import render_with_prefix

ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
ARM = "neutral_null"
N_SOURCES = 250
N_CONTINUATIONS = 25
N_CUTS = 6
CUT_SEED = 0
SAMPLE_SEED = 0
TOKEN_BUDGET = 16000


def count_tokens(text: str) -> int:
    """Same 4-chars-per-token estimate `driver.build_requests` used in step 3."""
    return len(text) // 4


def load_baseline_rows(parquet: Path):
    import pandas as pd  # local: only the build step needs it

    df = pd.read_parquet(parquet)
    base = df[df["direction"] == "baseline"].copy()
    # A prefix needs text to cut; the paper's own scoring drops unparsed rows.
    base = base[base["reasoning"].fillna("").str.len() > 0]
    base = base[base["estimate"].notna()]
    directional = df[df["direction"].isin(("below_good", "above_good"))]
    thresholds = {}
    for key, sub in directional.groupby("prompt_key"):
        vals = sorted(sub["threshold"].dropna().unique())
        if len(vals) != 1:
            raise ValueError(f"{key}: expected one threshold, found {vals}")
        thresholds[key] = float(vals[0])
    return base, thresholds


def sample_sources(base, thresholds, *, n_total: int, seed: int) -> list[dict]:
    """Uniform over prompt_keys: n_total // k each, remainder to the first
    keys in sorted order. Rows are sorted by rollout_id before drawing so the
    draw is a function of (parquet contents, seed) only."""
    keys = sorted(thresholds)
    per_key = {k: n_total // len(keys) for k in keys}
    for k in keys[: n_total - sum(per_key.values())]:
        per_key[k] += 1
    rng = random.Random(seed)
    table = _threshold_prompts()
    out = []
    for key in keys:
        rows = base[base["prompt_key"] == key].sort_values("rollout_id")
        if len(rows) < per_key[key]:
            raise ValueError(f"{key}: only {len(rows)} baseline rows, need {per_key[key]}")
        picked = rng.sample(list(rows.index), per_key[key])
        for idx in picked:
            r = rows.loc[idx]
            prompt = str(r["prompt"])
            if table is not None and table[key]["baseline"] != prompt:
                raise ValueError(f"{key}: cached baseline prompt != THRESHOLD_PROMPTS baseline")
            out.append(
                {
                    "source_id": str(r["rollout_id"]),
                    "prompt_key": key,
                    "direction": "baseline",
                    "threshold": thresholds[key],
                    "prompt": prompt,
                    "reasoning": str(r["reasoning"]),
                    "answer": str(r["answer"]) if r["answer"] is not None else "",
                    "estimate": float(r["estimate"]),
                }
            )
    return out


def build_requests(sources: list[dict]) -> tuple[list[dict], list[str]]:
    requests, skipped = [], []
    for src in sources:
        plan = plan_cuts(
            src["reasoning"], rollout_id=src["source_id"],
            n_cuts=N_CUTS, seed=CUT_SEED, jitter=True,
        )
        if len(plan["cuts"]) < 3 or plan["cuts"][1]["t"] == 0:
            skipped.append(src["source_id"])
            continue
        cut = plan["cuts"][1]  # the t ≈ 0.2 cut, as in step 3
        prefix_tokens = count_tokens(cut["prefix"])
        max_tokens = max(256, min(TOKEN_BUDGET, TOKEN_BUDGET - prefix_tokens))
        body = _continuation_body(
            render_with_prefix(src["prompt"], cut["prefix"]),
            model=MODEL, n=N_CONTINUATIONS, max_tokens=max_tokens,
            temperature=1.0, seed=None,
        )
        discriminator = [src["source_id"], cut["t"], ARM, 0]
        cell_id = f"{src['prompt_key']}|baseline|{src['threshold']:g}"
        requests.append(
            {
                "req_id": _req_id(body, MODEL, "continue", discriminator),
                "kind": "continue",
                "path": "/v1/completions",
                "body": body,
                "meta": {
                    "source_id": src["source_id"],
                    "cell_id": cell_id,
                    "prompt_key": src["prompt_key"],
                    "direction": "baseline",
                    "threshold": src["threshold"],
                    "arm": ARM,
                    "t": cut["t"],
                    "char_offset": cut["char_offset"],
                    "frac_sentences": cut["frac_sentences"],
                    "n_sentences": plan["n_sentences"],
                    "prefix_tokens_est": prefix_tokens,
                    "chunk_offset": 0,
                },
            }
        )
    return requests, skipped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet", type=Path, default=ART / "qwen3.5-35_master.parquet")
    b.add_argument("--sources-out", type=Path, default=ART / "step4d_sources.jsonl")
    b.add_argument("--out", type=Path, default=ART / "step4d_requests.jsonl")
    args = p.parse_args(argv)

    base, thresholds = load_baseline_rows(args.parquet)
    sources = sample_sources(base, thresholds, n_total=N_SOURCES, seed=SAMPLE_SEED)
    requests, skipped = build_requests(sources)
    _write_jsonl(args.sources_out, sources)
    _write_jsonl(args.out, requests)

    # Sanity.
    assert len(requests) == len({r["req_id"] for r in requests}), "duplicate req_ids"
    for r in requests:
        user = r["body"]["prompt"].split("<|im_end|>")[0]
        low = user.lower()
        assert "threshold" not in low and "bet" not in low, r["meta"]["source_id"]
        assert r["body"]["n"] == N_CONTINUATIONS
    ex = requests[len(requests) // 2]
    print(f"{len(sources)} sources -> {args.sources_out}")
    print(f"{len(requests)} requests ({sum(r['body']['n'] for r in requests):,} continuations) -> {args.out}")
    print("per prompt_key:", dict(sorted(Counter(r["meta"]["prompt_key"] for r in requests).items())))
    if skipped:
        print(f"skipped (too short for a t>0 second cut): {skipped}")
    fr = [r["meta"]["frac_sentences"] for r in requests]
    print(f"frac_sentences at kept cut: mean {sum(fr)/len(fr):.3f}, min {min(fr):.3f}, max {max(fr):.3f}")
    print("no 'threshold'/'bet' in any user turn: OK")
    print("\n--- example user turn ---")
    print(ex["body"]["prompt"].split("<|im_end|>")[0][len("<|im_start|>user\n"):])
    print("--- example prefix tail (repr) ---")
    print(repr(ex["body"]["prompt"][-200:]))
    print("--- example meta ---")
    print(json.dumps(ex["meta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
