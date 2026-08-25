#!/usr/bin/env python3
"""Estimate-judge for Donation Bet rollouts.

Extracts the model's final numeric estimate from free-text CoT+answer using
the paper's own judge prompt templates
(``THRESHOLD_PROMPTS[prompt_key]["judge_prompt"]``, imported from
``external/value_leakage/shared/prompts`` -- never copy-pasted) and its exact
``<final_estimate>`` tag parser (``shared.runner._parse_tagged_estimate``).

Judge backend is one small LLM call per rollout, configurable via env:
    JUDGE_BACKEND=anthropic (default) | openrouter
    JUDGE_MODEL=<model id>   default: claude-haiku-4-5-20251001 (anthropic)
                                       openai/gpt-4.1-mini      (openrouter)
    ANTHROPIC_API_KEY / OPENROUTER_API_KEY as required by the backend.

This intentionally does NOT reuse shared.runner's own judge machinery
(apply_judge / batch_extract_estimates): that path only knows how to build an
OpenAI-backend sender (``_infer_judge_backend`` raises for anything else) and
is wired to their on-disk JsonlJudgeCache layout. Reusing their judge PROMPT
and PARSER (the two things that actually determine correctness) while
supplying our own transport avoids both problems without forking any parsing
logic.

Standalone usage -- judge a full rollout file written by run_screen.py:
    python judge.py --in rollouts.jsonl --out rollouts.judged.jsonl
    python judge.py --in rollouts.jsonl --out /tmp/d.jsonl --dry-run

Rows that already carry a non-null "estimate" (run_screen.py fills this in
for baseline rows as a side effect of computing the median threshold) are
left untouched unless --force is passed, so re-running this after
run_screen.py only pays for judging the direction rows.

Library usage -- run_screen.py imports judge_texts_async directly (it's
already inside an asyncio event loop, so it can't use the sync wrapper) to
judge baselines before it can compute the per-question median threshold:
    from judge import judge_texts_async
    estimates = await judge_texts_async([(prompt_key, answer_text), ...])
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

_EXTERNAL_REPO = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

from shared.prompts import THRESHOLD_PROMPTS  # noqa: E402
from shared.runner import _parse_tagged_estimate  # noqa: E402

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"

# Matches the paper's own ESTIMATE_JUDGE_CONFIG (shared/runner.py) apart from
# backend/model, which we make configurable instead of hardcoding their
# claude-sonnet-4-6.
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 1024
JUDGE_MAX_CONCURRENT = 32


def _judge_backend_and_model():
    backend = os.environ.get("JUDGE_BACKEND", "anthropic").lower()
    if backend == "anthropic":
        model = os.environ.get("JUDGE_MODEL", DEFAULT_ANTHROPIC_MODEL)
    elif backend == "openrouter":
        model = os.environ.get("JUDGE_MODEL", DEFAULT_OPENROUTER_MODEL)
    else:
        raise ValueError(
            f"Unknown JUDGE_BACKEND={backend!r}; use 'anthropic' or 'openrouter'"
        )
    return backend, model


def _fabricate_judge_answer(prompt_key, rng):
    """Dry-run stand-in for a real judge call: emit a plausible
    <final_estimate> tag with no network call, so the whole pipeline
    (including median-threshold computation) is exercisable offline.

    NOT self-consistent with any fabricated rollout answer text -- see
    run_screen.py's --dry-run docstring for why that's an intentional scope
    cut (this validates plumbing, not statistics).
    """
    is_days = THRESHOLD_PROMPTS[prompt_key]["judge_prompt"].startswith(
        "I asked a model to estimate a duration"
    )
    value = rng.randint(1, 400) if is_days else rng.randint(1, 10_000_000)
    return f"<final_estimate>{value}</final_estimate>"


async def _judge_one_anthropic(client, model, rendered_prompt):
    resp = await client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        messages=[{"role": "user", "content": rendered_prompt}],
    )
    text_blocks = [b for b in resp.content if b.type == "text"]
    return text_blocks[0].text if text_blocks else ""


async def _judge_one_openrouter(client, model, rendered_prompt):
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        messages=[{"role": "user", "content": rendered_prompt}],
    )
    return resp.choices[0].message.content or ""


async def judge_texts_async(items, *, dry_run=False, seed=0):
    """``items``: list of (prompt_key, answer_text).

    Returns a list of float|None aligned 1:1 with ``items``, using the
    paper's own judge template (selected per-row by prompt_key) and tag
    parser -- so a "number" question and a "days" question in the same batch
    are each judged with their own template automatically.
    """
    rng = random.Random(seed)

    if dry_run:
        return [
            _parse_tagged_estimate(_fabricate_judge_answer(prompt_key, rng))
            for prompt_key, _answer_text in items
        ]

    backend, model = _judge_backend_and_model()
    rendered = [
        THRESHOLD_PROMPTS[prompt_key]["judge_prompt"].format(llm_text=text)
        for prompt_key, text in items
    ]

    semaphore = asyncio.Semaphore(JUDGE_MAX_CONCURRENT)
    if backend == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic()

        async def judge_one(p):
            return await _judge_one_anthropic(client, model, p)
    else:
        import openai

        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        async def judge_one(p):
            return await _judge_one_openrouter(client, model, p)

    async def guarded(p):
        async with semaphore:
            return await judge_one(p)

    raw_answers = await asyncio.gather(*(guarded(p) for p in rendered))
    return [_parse_tagged_estimate(a) for a in raw_answers]


def judge_texts(items, *, dry_run=False, seed=0):
    """Sync wrapper for callers not already inside an event loop (e.g. this
    module's own CLI). run_screen.py must NOT use this -- it's already
    running inside asyncio.run() and nesting a second one raises."""
    return asyncio.run(judge_texts_async(items, dry_run=dry_run, seed=seed))


# --- CLI: judge a full rollout JSONL written by run_screen.py ---


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


async def _main_async(args):
    rows = _load_jsonl(args.infile)

    to_judge_idx = [
        i for i, row in enumerate(rows)
        if args.force or row.get("estimate") is None
    ]
    skipped = len(rows) - len(to_judge_idx)

    items = [(rows[i]["prompt_key"], rows[i]["answer"]) for i in to_judge_idx]
    estimates = await judge_texts_async(items, dry_run=args.dry_run, seed=args.seed)
    for i, est in zip(to_judge_idx, estimates):
        rows[i]["estimate"] = est

    _write_jsonl(args.outfile, rows)

    n_parsed = sum(e is not None for e in estimates)
    n_unparsed = len(estimates) - n_parsed
    print(
        f"Judged {len(to_judge_idx)} rows ({skipped} already had an estimate "
        f"and were skipped; pass --force to re-judge everything): "
        f"{n_parsed} parsed, {n_unparsed} UNKNOWN/unparseable."
    )
    print(f"Wrote {args.outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", required=True,
                        help="rollout JSONL written by run_screen.py")
    parser.add_argument("--out", dest="outfile", required=True,
                        help="output path (adds/overwrites the estimate field per row)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fabricate judge answers instead of calling a real judge model")
    parser.add_argument("--force", action="store_true",
                        help="re-judge rows that already have a non-null estimate")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --dry-run fabrication")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
