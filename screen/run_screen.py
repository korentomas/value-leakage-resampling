#!/usr/bin/env python3
"""Donation Bet screen: sample Qwen3.5-35B-A3B / Qwen3.6-35B-A3B via a local
vLLM OpenAI-compatible endpoint, reproducing external/value_leakage's
``main_experiment_accurate`` config -- 9 questions x (n_baseline=100 +
n_per_threshold=100 x 2 directions), single median threshold computed from
each question's own baseline distribution.

Every prompt template, the experiment's question list/sample sizes, and the
threshold-computation logic are IMPORTED from external/value_leakage (never
copy-pasted), so this can't silently drift from the paper's actual config.

Phases (see README.md for the full run order including judge.py/bias.py):
  A. sample n_baseline baseline rollouts per question
  B. judge those baselines (judge.judge_texts_async) and compute each
     question's median threshold via the paper's own shared.runner
     ._compute_thresholds -- this is a real data dependency, not a design
     choice: the direction prompts in phase C need the threshold filled in
     before they can be built at all
  C. sample n_per_threshold below_good + above_good rollouts per question at
     that threshold
  D. write everything to one JSONL. Baseline rows carry their judged
     estimate as a side effect of phase B; direction rows get estimate=null
     until judge.py's standalone pass (a separate step -- keeps rollout
     sampling and judge-model calls independently retriable).

Usage:
    python run_screen.py --model-key qwen3.5-35 \\
        --served-model Qwen/Qwen3.5-35B-A3B \\
        --base-url http://localhost:8000/v1 \\
        --out rollouts_qwen3.5-35.jsonl

    # smoke test, no GPU/server/judge-model API key needed:
    python run_screen.py --dry-run --model-key qwen3.5-35 \\
        --n-baseline 5 --n-per-threshold 5 --out /tmp/dry.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_EXTERNAL_REPO = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
if str(_EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_REPO))

import pandas as pd  # noqa: E402

from shared.prompts import THRESHOLD_PROMPTS  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.runner import _compute_thresholds  # noqa: E402

from judge import judge_texts_async  # noqa: E402  (screen/judge.py, same dir)

DEFAULT_EXPERIMENT = "main_experiment_accurate"

# Paper's actual sampling config for qwen3.5-35 / qwen3.6-35
# (external/value_leakage/shared/models.py _TINKER_DEFAULTS; neither model
# key overrides temperature or max_tokens). This is 16000, NOT the 32000
# some *other* Tinker model entries use -- see README.md "Deviations".
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TOP_P = 1.0  # Tinker's SamplingParams call passes only
                      # max_tokens/temperature/stop -- no top_p/top_k, i.e.
                      # unrestricted nucleus/top-k. 1.0 is vLLM's matching
                      # "disabled" value.


def _split_reasoning(content: str, reasoning_content: str):
    """Return (answer, reasoning) robustly across vLLM reasoning-parser states.

    The Qwen3.5 chat template prefills ``<think>\n`` in the prompt, so the
    completion has no opening tag -- vLLM's qwen3 parser mis-handles this in
    both directions (vllm-project/vllm #38894, #35221): reasoning may arrive
    in reasoning_content (parser worked), embedded in content around
    ``</think>`` (parser off/missed), or content may be empty with EVERYTHING
    in reasoning_content (inverse bug / truncation). Normalise all three.
    """
    if reasoning_content and content and "</think>" not in content:
        return content, reasoning_content  # parser behaved
    text = content if "</think>" in content else (reasoning_content or content)
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        reasoning = head.split("<think>")[-1].strip("\n")
        return tail.lstrip("\n"), reasoning
    if reasoning_content and not content:
        # truncated inside the think block: reasoning only, no answer
        return "", reasoning_content
    return text, ""


async def _chat_once(client, args, prompt, retries=5):
    delay = 1.0
    last_exc = None
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=args.served_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            msg = resp.choices[0].message
            answer, reasoning = _split_reasoning(
                msg.content or "", getattr(msg, "reasoning_content", None) or ""
            )
            return {"prompt": prompt, "answer": answer, "reasoning": reasoning}
        except Exception as e:  # noqa: BLE001 -- broad on purpose, see retry loop
            last_exc = e
            print(
                f"  request failed (attempt {attempt + 1}/{retries}): "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"chat completion failed after {retries} attempts") from last_exc


async def run(args):
    experiment = THRESHOLD_EXPERIMENTS[args.experiment]
    prompt_keys = experiment["prompts"]
    n_baseline = args.n_baseline if args.n_baseline is not None else experiment["n_baseline"]
    n_per_threshold = (
        args.n_per_threshold if args.n_per_threshold is not None
        else experiment["n_per_threshold"]
    )
    threshold_spec = experiment["thresholds"]

    sampling = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }

    client = None
    if not args.dry_run:
        import openai

        client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Crash-salvage log: every completion is appended (and flushed) to
    # <out>.raw.jsonl the moment it arrives, so a mid-run crash of this
    # process (or the pod) loses nothing already sampled. The assembled
    # <out> file at the end stays the canonical artifact.
    raw_path = args.out + ".raw.jsonl"
    raw_lock = asyncio.Lock()
    raw_fh = None if args.dry_run else open(raw_path, "a")

    async def sample(prompt):
        if args.dry_run:
            # See judge.py's _fabricate_judge_answer docstring: this text is
            # NOT read by the dry-run judge, so it doesn't need to look like
            # a real completion -- dry-run validates plumbing, not stats.
            return {
                "prompt": prompt,
                "answer": "[DRY_RUN placeholder answer]",
                "reasoning": "[DRY_RUN placeholder reasoning]",
            }
        async with semaphore:
            result = await _chat_once(client, args, prompt)
        async with raw_lock:
            raw_fh.write(json.dumps({"ts": time.time(), **result}) + "\n")
            raw_fh.flush()
        return result

    async def gather_with_progress(tasks, label):
        # Same semantics as asyncio.gather(*tasks) (ordered results), plus
        # periodic machine-readable "PROGRESS <label> <done>/<total>" lines
        # on stdout for the live dashboard (watch.py) to parse.
        total = len(tasks)
        done = 0
        results = [None] * total

        async def runner(i, coro):
            nonlocal done
            results[i] = await coro
            done += 1
            if done % 5 == 0 or done == total:
                print(f"PROGRESS {label} {done}/{total}", flush=True)

        await asyncio.gather(*(runner(i, c) for i, c in enumerate(tasks)))
        return results

    # --- Phase A: baseline sampling, all questions ---
    print(f"Phase A: sampling {n_baseline} baselines x {len(prompt_keys)} questions...")
    baseline_meta = []
    baseline_tasks = []
    for pk in prompt_keys:
        prompt_text = THRESHOLD_PROMPTS[pk]["baseline"]
        for i in range(n_baseline):
            baseline_meta.append({"prompt_key": pk, "replicate": i})
            baseline_tasks.append(sample(prompt_text))
    baseline_results = await gather_with_progress(baseline_tasks, "baseline")

    # --- Phase B: judge baselines, compute each question's median threshold ---
    print("Phase B: judging baselines to compute median thresholds...")
    judge_items = [
        (m["prompt_key"], r["answer"]) for m, r in zip(baseline_meta, baseline_results)
    ]
    baseline_estimates = await judge_texts_async(
        judge_items, dry_run=args.dry_run, seed=args.seed,
    )

    estimates_by_pk = {pk: [] for pk in prompt_keys}
    for m, est in zip(baseline_meta, baseline_estimates):
        estimates_by_pk[m["prompt_key"]].append(est)

    thresholds = {}
    for pk in prompt_keys:
        series = pd.Series(estimates_by_pk[pk])
        # threshold_spec is the single-median spec (pct_range=(50,50), n=1)
        # for every experiment in THRESHOLD_EXPERIMENTS as of this writing;
        # _compute_thresholds still returns a list for generality, so take
        # the one value.
        thresholds[pk] = _compute_thresholds(series, threshold_spec)[0]
        n_parsed = int(series.notna().sum())
        print(f"  {pk}: threshold={thresholds[pk]} ({n_parsed}/{len(series)} baselines parsed)")
        if n_parsed == 0:
            raise RuntimeError(
                f"{pk}: 0/{len(series)} baseline estimates parsed -- cannot compute a "
                "threshold. Check the judge backend/model and JUDGE_BACKEND/JUDGE_MODEL "
                "env vars (see judge.py)."
            )

    # --- Phase C: direction sampling at the computed thresholds ---
    print(
        f"Phase C: sampling {n_per_threshold} x 2 directions x {len(prompt_keys)} "
        "questions..."
    )
    direction_meta = []
    direction_tasks = []
    for pk in prompt_keys:
        t = thresholds[pk]
        prompt_set = THRESHOLD_PROMPTS[pk]
        for direction, template_key in (
            ("below_good", "below_good_template"),
            ("above_good", "above_good_template"),
        ):
            # Comma-formatted threshold, matching shared.runner
            # ._run_directions_for_prompt exactly (template.format(threshold=f"{t:,}")).
            prompt_text = prompt_set[template_key].format(threshold=f"{t:,}")
            for i in range(n_per_threshold):
                direction_meta.append(
                    {"prompt_key": pk, "direction": direction, "threshold": t, "replicate": i}
                )
                direction_tasks.append(sample(prompt_text))
    direction_results = await gather_with_progress(direction_tasks, "directions")

    # --- Assemble + write ---
    rows = []
    for m, r, est in zip(baseline_meta, baseline_results, baseline_estimates):
        rows.append({
            "model_key": args.model_key,
            "served_model": args.served_model,
            "experiment": args.experiment,
            "prompt_key": m["prompt_key"],
            "direction": "baseline",
            "threshold": None,
            "replicate": m["replicate"],
            "prompt": r["prompt"],
            "reasoning": r["reasoning"],
            "answer": r["answer"],
            "estimate": est,
            "sampling": sampling,
            "dry_run": args.dry_run,
        })
    for m, r in zip(direction_meta, direction_results):
        rows.append({
            "model_key": args.model_key,
            "served_model": args.served_model,
            "experiment": args.experiment,
            "prompt_key": m["prompt_key"],
            "direction": m["direction"],
            "threshold": m["threshold"],
            "replicate": m["replicate"],
            "prompt": r["prompt"],
            "reasoning": r["reasoning"],
            "answer": r["answer"],
            "estimate": None,  # filled by judge.py
            "sampling": sampling,
            "dry_run": args.dry_run,
        })

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(
        f"\nWrote {len(rows)} rollouts to {args.out} "
        f"({len(baseline_meta)} baseline, {len(direction_meta)} direction)."
    )
    print(f"Next: python judge.py --in {args.out} --out {args.out.replace('.jsonl', '.judged.jsonl')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True,
                        help="screen registry key, e.g. qwen3.5-35 or qwen3.6-35 "
                             "(recorded in output rows; bias.py uses it to pick the gate)")
    parser.add_argument("--served-model", default="Qwen/Qwen3.5-35B-A3B",
                        help="HF repo id passed as the OpenAI `model` field / to vLLM's serve.sh")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY", help="vLLM doesn't check this; any string works")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--n-baseline", type=int, default=None,
                        help="override experiment's n_baseline (default: 100, from THRESHOLD_EXPERIMENTS)")
    parser.add_argument("--n-per-threshold", type=int, default=None,
                        help="override experiment's n_per_threshold (default: 100)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --dry-run judge fabrication")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="fabricate rollouts and judge estimates; no server, no judge API key needed")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
