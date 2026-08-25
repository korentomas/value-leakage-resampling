#!/usr/bin/env python3
"""Judge calibration (STEP2.md run-order item 2i, prereg 2026-08-18).

Judges a stratified sample of screen answers with the paper's reference
judge (claude-sonnet-4.6, temp 0) AND each cheap candidate from the
trustworthy ladder; adopts the cheapest candidate whose parsed estimates
exactly agree with the reference on >= 99.5% of rows.

    JUDGE_BACKEND=openrouter OPENROUTER_API_KEY=... \
    python calibrate_judge.py --in ../artifacts/screen_qwen3.5-35_fp8.jsonl \
        --out ../artifacts/judge_calibration.json --n 300

Exit 0 always (the adopted model is in the output JSON; empty = none passed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import judge_texts_async  # noqa: E402

REFERENCE = "anthropic/claude-sonnet-4.6"
# cheapest-first ladder, prereg'd (major-lab only per TK)
CANDIDATES = [
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4.1-mini",
    "anthropic/claude-haiku-4.5",
]
AGREEMENT_BAR = 0.995


def load_items(path, n, seed):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("direction") != "baseline" and r.get("answer")]
    rng = random.Random(seed)
    # stratify: proportional draw per prompt_key, seeded shuffle then round-robin
    by_pk = {}
    for r in rows:
        by_pk.setdefault(r["prompt_key"], []).append(r)
    for v in by_pk.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < min(n, len(rows)):
        for v in by_pk.values():
            if i < len(v):
                picked.append(v[i])
                if len(picked) >= min(n, len(rows)):
                    break
        i += 1
    return [(r["prompt_key"], r["answer"]) for r in picked]


def agree(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return float(a) == float(b)


async def run(args):
    items = load_items(args.infile, args.n, args.seed)
    print(f"calibrating on {len(items)} answers "
          f"({len(set(pk for pk, _ in items))} questions)", flush=True)

    os.environ["JUDGE_BACKEND"] = "openrouter"

    os.environ["JUDGE_MODEL"] = REFERENCE
    ref = await judge_texts_async(items, seed=args.seed)
    n_ref_parsed = sum(v is not None for v in ref)
    print(f"reference {REFERENCE}: {n_ref_parsed}/{len(items)} parsed", flush=True)

    report = {"reference": REFERENCE, "n_items": len(items),
              "n_ref_parsed": n_ref_parsed, "candidates": {}, "adopted": None}

    for cand in CANDIDATES:
        os.environ["JUDGE_MODEL"] = cand
        out = await judge_texts_async(items, seed=args.seed)
        agreements = [agree(r, c) for r, c in zip(ref, out)]
        rate = sum(agreements) / len(agreements)
        disagreements = [
            {"prompt_key": items[i][0], "ref": ref[i], "cand": out[i],
             "answer_tail": items[i][1][-200:]}
            for i, ok in enumerate(agreements) if not ok
        ][:8]
        report["candidates"][cand] = {
            "agreement": rate,
            "n_parsed": sum(v is not None for v in out),
            "n_disagree": len(agreements) - sum(agreements),
            "examples": disagreements,
        }
        print(f"candidate {cand}: agreement {rate:.4f} "
              f"({len(agreements) - sum(agreements)} disagree)", flush=True)
        if rate >= AGREEMENT_BAR and report["adopted"] is None:
            report["adopted"] = cand

    if report["adopted"] is None:
        report["adopted"] = REFERENCE
        print(f"no candidate passed {AGREEMENT_BAR:.1%}; adopting reference "
              f"(cost consequences per STEP2.md)", flush=True)
    print("ADOPTED:", report["adopted"], flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
