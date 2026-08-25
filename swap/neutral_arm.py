"""Neutral (no-bet) third arm for the prefix-swap design.

Derives continuation requests from an existing step-3 request file so that
every neutral-arm continuation resumes the *byte-identical* prefix that the
orig and swap arms used: the orig-arm request's prompt is split into
(template header rendered from the source's prompt, prefix), and the prefix
is re-rendered under the paper's `baseline` prompt for the same question.

    dep_orig_vs_neutral(t) = P(favored | orig) - P(favored | neutral)
    dep_swap_vs_neutral(t) = P(favored | neutral) - P(favored | swap)

Together with the two-arm dep they separate "the answer is locked" (all three
arms agree) from "the model follows the prefix over a contradicting prompt"
(neutral tracks orig while swap lags).

Usage:
    python neutral_arm.py build --requests ../artifacts/step3_requests.jsonl \
        --sources ../artifacts/step3_sources.jsonl \
        --out ../artifacts/step4_neutral_requests.jsonl --shards 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from driver import _continuation_body, _req_id, _threshold_prompts, iter_jsonl, load_sources
from template import render_generation_prompt, render_with_prefix


def baseline_prompt(prompt_key: str) -> str:
    table = _threshold_prompts()
    if table is None:
        raise RuntimeError("external/value_leakage is not importable")
    return table[prompt_key]["baseline"]


def build_neutral_requests(requests: list[dict], sources_by_id: dict) -> list[dict]:
    out = []
    for req in requests:
        meta = req["meta"]
        if req["kind"] != "continue" or meta["arm"] != "orig" or meta["source_id"] is None:
            continue
        src = sources_by_id[meta["source_id"]]
        header = render_generation_prompt(src.prompt)
        prompt = req["body"]["prompt"]
        if not prompt.startswith(header):
            raise ValueError(f"orig prompt does not start with source header: {req['req_id']}")
        prefix = prompt[len(header):]
        neutral_user = baseline_prompt(meta["prompt_key"])
        body = _continuation_body(
            render_with_prefix(neutral_user, prefix),
            model=req["body"]["model"],
            n=req["body"]["n"],
            max_tokens=req["body"]["max_tokens"],
            temperature=req["body"]["temperature"],
            seed=req["body"].get("seed"),
        )
        # Same discriminator layout as driver.build_requests, arm renamed.
        discriminator = [meta["source_id"], meta["t"], "neutral", meta["chunk_offset"]]
        out.append(
            {
                "req_id": _req_id(body, req["body"]["model"], "continue", discriminator),
                "kind": "continue",
                "path": req["path"],
                "body": body,
                "meta": {**meta, "arm": "neutral", "derived_from": req["req_id"]},
            }
        )
    deduped = {}
    for r in out:
        deduped.setdefault(r["req_id"], r)
    return list(deduped.values())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--requests", type=Path, required=True)
    b.add_argument("--sources", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--shards", type=int, default=1)
    args = p.parse_args(argv)

    sources = {s.source_id: s for s in load_sources(args.sources)}
    requests = list(iter_jsonl(args.requests))
    neutral = build_neutral_requests(requests, sources)

    # Sanity: each neutral request's prefix equals its orig sibling's prefix.
    by_id = {r["req_id"]: r for r in requests}
    for r in neutral:
        orig = by_id[r["meta"]["derived_from"]]
        src = sources[r["meta"]["source_id"]]
        p_orig = orig["body"]["prompt"][len(render_generation_prompt(src.prompt)):]
        p_neut = r["body"]["prompt"][len(render_generation_prompt(baseline_prompt(r["meta"]["prompt_key"]))):]
        assert p_orig == p_neut, r["req_id"]
        assert "bet" not in baseline_prompt(r["meta"]["prompt_key"]).lower()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in neutral:
            f.write(json.dumps(r) + "\n")
    if args.shards > 1:
        stem = args.out.with_suffix("")
        for i in range(args.shards):
            with Path(f"{stem}_n{i+1}.jsonl").open("w") as f:
                for r in neutral[i::args.shards]:
                    f.write(json.dumps(r) + "\n")
    n_src = len({r["meta"]["source_id"] for r in neutral})
    n_samples = sum(r["body"]["n"] for r in neutral)
    print(f"{len(neutral):,} neutral requests ({n_src} sources, {n_samples:,} continuations) -> {args.out}")
    print("example prompt tail (repr):", repr(neutral[0]["body"]["prompt"][-200:]))
    print("example user turn (first 160 chars):", repr(neutral[0]["body"]["prompt"][:160]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
