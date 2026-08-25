"""Step 4G: small generalisation run of the three-arm design on Qwen3.6-35B-A3B.

The paper's other open-weight model (bias 0.27, 65.5% deniers). Same
instrument as steps 3 / 4A, scaled down:

  * 100 sources drawn (seed 0) uniformly over the 18 (prompt_key, direction)
    cells of the paper's cached Qwen3.6 intervention rollouts (5 or 6 per
    cell), non-empty reasoning and a parsed estimate;
  * each cut with the step-3 rule (`plan_cuts`, n_cuts=6, seed 0, jitter);
    only the second (t ≈ 0.2) and the last (t = 1.0, whole CoT) cuts kept;
  * arms orig + swap at both cuts, neutral (paper's no-bet `baseline`
    prompt, as `neutral_arm.py`) at t ≈ 0.2 only;
  * N = 25, temperature 1, max_tokens = 16000 − prefix tokens (4 chars/tok
    estimate, as in step 3), Qwen/Qwen3.6-35B-A3B-FP8 on /v1/completions.

Expected: 100 × (2 × 2 + 1) = 500 requests, 12,500 continuations.

Thresholds are Qwen3.6's *own* baseline medians (the parquet's `threshold`
column), not Qwen3.5's. The swap arm's user turn is the *cached* prompt of
the opposite direction for the same question, not `driver.build_counter_
prompt`: the paper renders thresholds with thousands separators
(`shared/runner.py`: ``template.format(threshold=f"{t:,}")``) and
`build_counter_prompt` does not, so the two differ for every threshold
>= 1000. The cached prompt is byte-identical to what the source rollout's
opposite condition actually saw.

Also written, from the cache alone: the t = 0 orig/swap cell counts
(`driver.counts_from_cache`) and the per-cell baseline (no-bet) counts
(`driver.baseline_counts_from_cache`), in the layout step 3 / 4A used.

Usage (from swap/):
    ../external/value_leakage/.venv/bin/python qwen36_gen.py build
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cuts import plan_cuts
from driver import (
    DIRECTIONS,
    OPPOSITE,
    _continuation_body,
    _req_id,
    _threshold_prompts,
    _write_jsonl,
    baseline_counts_from_cache,
    build_counter_prompt,
    counts_from_cache,
)
from template import render_with_prefix

ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
PATH = "/v1/completions"
N_SOURCES = 100
N_CONTINUATIONS = 25
N_CUTS = 6
CUT_SEED = 0
SAMPLE_SEED = 0
TOKEN_BUDGET = 16000
KEEP_CUTS = {1: ("orig", "swap", "neutral"), N_CUTS - 1: ("orig", "swap")}


def count_tokens(text: str) -> int:
    """Same 4-chars-per-token estimate `driver.build_requests` used in step 3."""
    return len(text) // 4


def paper_threshold_str(threshold: float) -> str:
    """The paper's rendering: `f"{t:,}"` with t an int when integer-valued."""
    f = float(threshold)
    return f"{int(f):,}" if f.is_integer() else f"{f:,}"


def load_rows(parquet: Path):
    import pandas as pd  # local: only the build step needs it

    df = pd.read_parquet(parquet)
    d = df[df["direction"].isin(DIRECTIONS)].copy()
    d = d[d["reasoning"].fillna("").str.len() > 0]
    d = d[d["estimate"].notna()]

    thresholds, prompts = {}, {}
    table = _threshold_prompts()
    if table is None:
        raise RuntimeError("external/value_leakage is not importable")
    for key, sub in d.groupby("prompt_key"):
        vals = sorted(sub["threshold"].dropna().unique())
        if len(vals) != 1:
            raise ValueError(f"{key}: expected one threshold, found {vals}")
        thresholds[key] = float(vals[0])
        for direction in DIRECTIONS:
            ps = sub[sub["direction"] == direction]["prompt"].unique()
            if len(ps) != 1:
                raise ValueError(f"{key}/{direction}: {len(ps)} distinct cached prompts")
            prompts[key, direction] = str(ps[0])
            # The cached prompt is the paper's template with a comma-formatted
            # threshold; check that reading is right before relying on it.
            expect = table[key][f"{direction}_template"].format(
                threshold=paper_threshold_str(thresholds[key])
            )
            if expect != prompts[key, direction]:
                raise ValueError(f"{key}/{direction}: cached prompt != paper template render")
        base = df[(df["prompt_key"] == key) & (df["direction"] == "baseline")]["prompt"].unique()
        if len(base) != 1 or str(base[0]) != table[key]["baseline"]:
            raise ValueError(f"{key}: cached baseline prompt != THRESHOLD_PROMPTS baseline")
    return df, d, thresholds, prompts


def valid_plan(src_id: str, reasoning: str):
    plan = plan_cuts(reasoning, rollout_id=src_id, n_cuts=N_CUTS, seed=CUT_SEED, jitter=True)
    cuts = plan["cuts"]
    if len(cuts) != N_CUTS or cuts[1]["t"] == 0 or cuts[-1]["t"] != plan["n_sentences"]:
        return None
    return plan


def sample_sources(d, thresholds, prompts, *, n_total: int, seed: int) -> list[dict]:
    """Uniform over the 18 (prompt_key, direction) cells: n_total // 18 each,
    remainder to the first cells in sorted order. Within a cell the rows are
    sorted by rollout_id, shuffled with the seeded rng, and taken in order,
    skipping any rollout too short for a t > 0 second cut — so the draw is a
    function of (parquet contents, seed) only and always fills the quota."""
    cells = [(k, dr) for k in sorted(thresholds) for dr in DIRECTIONS]
    quota = {c: n_total // len(cells) for c in cells}
    for c in cells[: n_total - sum(quota.values())]:
        quota[c] += 1
    rng = random.Random(seed)
    out, skipped = [], []
    for key, direction in cells:
        rows = d[(d["prompt_key"] == key) & (d["direction"] == direction)].sort_values("rollout_id")
        order = list(rows.index)
        rng.shuffle(order)
        taken = 0
        for idx in order:
            if taken == quota[key, direction]:
                break
            r = rows.loc[idx]
            sid = str(r["rollout_id"])
            if valid_plan(sid, str(r["reasoning"])) is None:
                skipped.append(sid)
                continue
            out.append(
                {
                    "source_id": sid,
                    "prompt_key": key,
                    "direction": direction,
                    "threshold": thresholds[key],
                    "prompt": prompts[key, direction],
                    "reasoning": str(r["reasoning"]),
                    "counter_prompt": prompts[key, OPPOSITE[direction]],
                    "answer": str(r["answer"]) if r["answer"] is not None else "",
                    "estimate": float(r["estimate"]),
                }
            )
            taken += 1
        if taken < quota[key, direction]:
            raise ValueError(f"{key}/{direction}: only {taken} usable rows, need {quota[key, direction]}")
    return out, skipped


def build_requests(sources: list[dict]) -> list[dict]:
    table = _threshold_prompts()
    requests = []
    for src in sources:
        plan = valid_plan(src["source_id"], src["reasoning"])
        assert plan is not None, src["source_id"]
        cell_id = f"{src['prompt_key']}|{src['direction']}|{src['threshold']:g}"
        users = {
            "orig": src["prompt"],
            "swap": src["counter_prompt"],
            "neutral": table[src["prompt_key"]]["baseline"],
        }
        for cut_idx, arms in KEEP_CUTS.items():
            cut = plan["cuts"][cut_idx]
            prefix_tokens = count_tokens(cut["prefix"])
            max_tokens = max(256, min(TOKEN_BUDGET, TOKEN_BUDGET - prefix_tokens))
            orig_req_id = None
            for arm in arms:
                body = _continuation_body(
                    render_with_prefix(users[arm], cut["prefix"]),
                    model=MODEL, n=N_CONTINUATIONS, max_tokens=max_tokens,
                    temperature=1.0, seed=None,
                )
                discriminator = [src["source_id"], cut["t"], arm, 0]
                req_id = _req_id(body, MODEL, "continue", discriminator)
                if arm == "orig":
                    orig_req_id = req_id
                meta = {
                    "source_id": src["source_id"],
                    "cell_id": cell_id,
                    "prompt_key": src["prompt_key"],
                    "direction": src["direction"],
                    "threshold": src["threshold"],
                    "arm": arm,
                    "t": cut["t"],
                    "char_offset": cut["char_offset"],
                    "n_sentences": plan["n_sentences"],
                    "frac_sentences": cut["frac_sentences"],
                    "prefix_tokens_est": prefix_tokens,
                    "chunk_offset": 0,
                }
                if arm == "neutral":
                    meta["derived_from"] = orig_req_id
                requests.append(
                    {"req_id": req_id, "kind": "continue", "path": PATH, "body": body, "meta": meta}
                )
    deduped = {}
    for r in requests:
        deduped.setdefault(r["req_id"], r)
    return list(deduped.values())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet", type=Path, default=ART / "qwen3.6-35_master.parquet")
    b.add_argument("--sources-out", type=Path, default=ART / "step4g_sources.jsonl")
    b.add_argument("--out", type=Path, default=ART / "step4g_requests.jsonl")
    b.add_argument("--t0-out", type=Path, default=ART / "step4g_t0_counts.jsonl")
    b.add_argument("--baseline-out", type=Path, default=ART / "step4g_baseline_counts.jsonl")
    args = p.parse_args(argv)

    df, d, thresholds, prompts = load_rows(args.parquet)
    sources, skipped = sample_sources(d, thresholds, prompts, n_total=N_SOURCES, seed=SAMPLE_SEED)
    requests = build_requests(sources)

    # --- sanity ---------------------------------------------------------
    assert len(sources) == N_SOURCES
    assert len(requests) == len({r["req_id"] for r in requests}), "duplicate req_ids"
    expected = N_SOURCES * sum(len(a) for a in KEEP_CUTS.values())
    assert len(requests) == expected, (len(requests), expected)
    table = _threshold_prompts()
    fmt_mismatch = set()
    for s in sources:
        bcp = build_counter_prompt(s["prompt_key"], s["direction"], s["threshold"])
        if bcp != s["counter_prompt"]:
            fmt_mismatch.add(s["prompt_key"])
            assert bcp.replace(",", "") == s["counter_prompt"].replace(",", "")
    for r in requests:
        m = r["meta"]
        user = r["body"]["prompt"].split("<|im_end|>")[0]
        low = user.lower()
        if m["arm"] == "neutral":
            assert "threshold" not in low and "bet" not in low, r["req_id"]
        else:
            assert f"The threshold is {paper_threshold_str(m['threshold'])}." in user, r["req_id"]
        assert r["body"]["n"] == N_CONTINUATIONS and r["body"]["model"] == MODEL
        assert r["body"]["max_tokens"] == max(256, min(TOKEN_BUDGET, TOKEN_BUDGET - m["prefix_tokens_est"]))
    # neutral/orig/swap at the same (source, cut) share a byte-identical prefix
    by_key = defaultdict(dict)
    for r in requests:
        m = r["meta"]
        hdr_user = {"orig": None, "swap": None, "neutral": table[m["prompt_key"]]["baseline"]}[m["arm"]]
        prompt = r["body"]["prompt"]
        user = prompt.split("<|im_end|>")[0][len("<|im_start|>user\n"):]
        prefix = prompt[len(render_with_prefix(user, "")):]
        by_key[m["source_id"], m["t"]][m["arm"]] = prefix
    for (sid, t), arms in by_key.items():
        assert len(set(arms.values())) == 1, (sid, t)

    # --- cache-derived cells ---------------------------------------------
    t0 = counts_from_cache(df[df["direction"].isin(DIRECTIONS)])
    base = baseline_counts_from_cache(df)
    dep0 = []
    for cell in {c["cell_id"] for c in t0}:
        po = next(c for c in t0 if c["cell_id"] == cell and c["arm"] == "orig")
        ps = next(c for c in t0 if c["cell_id"] == cell and c["arm"] == "swap")
        dep0.append(po["k"] / po["n"] - ps["k"] / ps["n"])

    _write_jsonl(args.sources_out, sources)
    _write_jsonl(args.out, requests)
    _write_jsonl(args.t0_out, t0)
    _write_jsonl(args.baseline_out, base)

    # --- report ------------------------------------------------------------
    print(f"{len(sources)} sources -> {args.sources_out}")
    print("per (prompt_key, direction):",
          dict(sorted(Counter((s["prompt_key"], s["direction"]) for s in sources).items())))
    if skipped:
        print(f"skipped while drawing (too short for a t>0 second cut): {skipped}")
    print(f"{len(requests)} requests ({sum(r['body']['n'] for r in requests):,} continuations) -> {args.out}")
    by_arm_cut = Counter((r["meta"]["arm"], round(r["meta"]["frac_sentences"], 1) == 1.0) for r in requests)
    print("per arm x cut (False = t~0.2, True = t=1.0):", dict(sorted(by_arm_cut.items())))
    fr = [r["meta"]["frac_sentences"] for r in requests if r["meta"]["t"] != r["meta"]["n_sentences"]]
    print(f"frac_sentences at the early cut: mean {sum(fr)/len(fr):.3f}, min {min(fr):.3f}, max {max(fr):.3f}")
    mt = [r["body"]["max_tokens"] for r in requests]
    print(f"max_tokens: min {min(mt)}, median {sorted(mt)[len(mt)//2]}, max {max(mt)}")
    print(f"counter_prompt != driver.build_counter_prompt (comma formatting) for: {sorted(fmt_mismatch)}")
    print(f"{len(t0)} t=0 orig/swap cells -> {args.t0_out}; mean cell dep(0) = {sum(dep0)/len(dep0):.3f} (published 0.27)")
    print(f"{len(base)} baseline cells -> {args.baseline_out}; "
          + ", ".join(f"{c['cell_id'].split('|')[0][3:-9]}/{c['direction'][:5]} {c['k']}/{c['n']}" for c in base))
    for arm in ("orig", "swap", "neutral"):
        ex = next(r for r in requests if r["meta"]["arm"] == arm and r["meta"]["t"] != r["meta"]["n_sentences"])
        user = ex["body"]["prompt"].split("<|im_end|>")[0][len("<|im_start|>user\n"):]
        print(f"\n--- example {arm} (source {ex['meta']['source_id'][:12]}, t={ex['meta']['t']}/{ex['meta']['n_sentences']}) ---")
        print("user turn tail:", repr(user[-230:]))
        print("prefix tail:   ", repr(ex["body"]["prompt"][-160:]))
        print("meta:", json.dumps(ex["meta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
