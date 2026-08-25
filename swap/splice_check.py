"""Splice-fidelity smoke test (STEP2 item 1d, blocking).

The claim under test
--------------------
Swap assumption 3: our template rendering + `/v1/completions` prefill path
reproduces the model's own cached behaviour. If splicing a rollout's FULL
CoT back under its ORIGINAL prompt does not regenerate answers on the same
side of the threshold as the cached answer, every downstream dep(t) number
is measuring a rendering artefact, not value leakage — so this check gates
all continuation spend.

Design (preregistered in STEP2.md)
----------------------------------
Sample `--n-sources` (20) intervention rollouts from the master parquet,
uniformly, seeded. For each, splice the complete normalised reasoning —
`</think>` NOT closed, the model must close it and write the answer — after
`<think>\n` under the rollout's own prompt, and draw `--n-continuations`
(10) continuations at temperature 1.0.

Per rollout: the fraction of continuations whose estimate lands on the same
side of the threshold as the cached estimate. Aggregate: exact one-sided
sign test across sources that majority-side == cached-side, H0 p=0.5;
pass = binom_p < 0.01 (this is the one preregistered frequentist test in
the project). A pooled-continuations-vs-cached KS statistic and the
per-rollout fractions are reported as descriptive numbers.

Estimate extraction
-------------------
Cached answers carry no `<final_estimate>` tags (0/1636 in the parquet);
the cached `estimate` column was produced by the paper's LLM judge. To
compare like with like, the default extractor is the same instrument the
screen uses — `screen/judge.py::judge_texts_async` (paper's judge template
+ tag parser, haiku backend, needs ANTHROPIC_API_KEY; ~200 calls).
`--regex-judge` substitutes a first-number regex for offline runs: it
side-agrees with the cached judge on 97.2% of cached intervention rows, so
treat a borderline result under it as unresolved, not as a verdict.

Usage
-----
    # on the pod, real run
    python splice_check.py --base-url http://localhost:8000/v1 \
        --out splice_results.jsonl

    # plumbing check, no server, no API key
    python splice_check.py --dry-run

Continuation results are appended one line per source request (flushed), so
an interrupted run resumes where it stopped. Judge results are cached in
the same file. Dashboard parses the `PROGRESS splice <done>/<total>` lines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
from pathlib import Path

import pandas as pd

from cuts import make_prefix
from driver import (
    _Writer,
    _continuation_body,
    _post_one,
    _req_id,
    answer_cache_key,
    completed_ids,
    iter_jsonl,
    parse_tagged_estimate,
)
from template import render_with_prefix, split_completion

DEFAULT_MASTER = Path("../artifacts/qwen3.5-35_master.parquet")
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
TOKEN_BUDGET = 16000  # the paper's total sampling budget (shared/models.py)

# First plausible number in free text: optional sign/$, digit grouping,
# decimals, and `x 10^k` / `1e9` scientific forms. Only used with
# --regex-judge or --dry-run; the default instrument is the LLM judge.
_NUMBER_RE = re.compile(
    r"-?\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:x|×)\s*10\^?\d+|[eE][+-]?\d+)?"
)


# --- source sampling ------------------------------------------------------


def sample_sources(master_path: Path, n_sources: int, seed: int) -> pd.DataFrame:
    """Uniform seeded sample of eligible intervention rollouts."""
    df = pd.read_parquet(master_path)
    eligible = df[
        (df["direction"] != "baseline")
        & df["estimate"].notna()
        & df["threshold"].notna()
        & (df["reasoning"].fillna("").str.strip().str.len() > 0)
    ]
    if len(eligible) < n_sources:
        raise ValueError(
            f"only {len(eligible)} eligible rollouts in {master_path}, "
            f"need {n_sources}"
        )
    return eligible.sample(n=n_sources, random_state=seed).reset_index(drop=True)


# --- request construction -------------------------------------------------


def build_splice_requests(
    sources: pd.DataFrame, *, model: str, n_continuations: int
) -> list[dict]:
    """One /v1/completions request per source: full-CoT prefix, n samples.

    The prefix is the complete normalised reasoning (`make_prefix` at full
    length: markup stripped, trailing spaces removed so the prompt does not
    end mid-token). `</think>` is left open — closing it is part of the
    behaviour under test. `max_tokens` follows the driver's budget logic:
    the paper sampled under a 16000-token total cap, so the continuation
    keeps whatever of that budget the prefix has not already spent.
    """
    requests = []
    for _, row in sources.iterrows():
        reasoning = row["reasoning"]
        prefix = make_prefix(reasoning, len(reasoning))
        prompt = render_with_prefix(row["prompt"], prefix)
        prefix_tokens = len(prefix) // 4
        max_tokens = max(256, min(TOKEN_BUDGET, TOKEN_BUDGET - prefix_tokens))
        body = _continuation_body(
            prompt,
            model=model,
            n=n_continuations,
            max_tokens=max_tokens,
            temperature=1.0,
            seed=None,
        )
        discriminator = [row["rollout_id"], "splice_full_cot"]
        requests.append(
            {
                "req_id": _req_id(body, model, "continue", discriminator),
                "kind": "continue",
                "path": "/v1/completions",
                "body": body,
                "meta": {
                    "source_id": row["rollout_id"],
                    "prompt_key": row["prompt_key"],
                    "direction": row["direction"],
                    "threshold": float(row["threshold"]),
                    "cached_estimate": float(row["estimate"]),
                    "prefix_chars": len(prefix),
                    "prefix_tokens_est": prefix_tokens,
                },
            }
        )
    return requests


# --- execution ------------------------------------------------------------


async def run_splice_requests(
    requests: list[dict],
    results_path: Path,
    *,
    base_url: str,
    concurrency: int,
    timeout: float,
) -> None:
    """Execute pending requests with a PROGRESS line per completed source."""
    import httpx

    done = completed_ids(results_path)
    pending = [r for r in requests if r["req_id"] not in done]
    total = len(requests)
    finished = total - len(pending)
    print(f"PROGRESS splice {finished}/{total}", flush=True)
    if not pending:
        return

    writer = _Writer(results_path, fsync_every=1)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def worker(req):
            nonlocal finished
            async with sem:
                rec = await _post_one(
                    client, base_url, req, timeout=timeout, max_retries=5
                )
            await writer.write(rec)
            finished += 1
            status = "FAILED " + rec["error"][:80] if rec.get("error") else "ok"
            print(
                f"PROGRESS splice {finished}/{total}", flush=True
            )
            print(f"  source {req['meta']['source_id'][:12]} {status}", flush=True)

        try:
            await asyncio.gather(*(worker(r) for r in pending))
        finally:
            writer.close()


def fabricate_results(
    requests: list[dict], results_path: Path, *, n_continuations: int, seed: int
) -> None:
    """--dry-run stand-in for the server: echo the cached side with noise.

    Each fabricated continuation closes the think block and answers with a
    value on the cached estimate's side of the threshold (side-aware, not
    estimate-plus-noise: sources whose cached estimate sits on or near the
    threshold would flip sides under any multiplicative noise, failing the
    sign test for reasons that have nothing to do with plumbing); 10% flip
    to the opposite side and 5% never close `</think>` (exercising the
    unparsed path). Validates plumbing, not statistics.
    """
    rng = random.Random(seed + 1)
    done = completed_ids(results_path)
    writer = _Writer(results_path, fsync_every=1)
    for req in requests:
        if req["req_id"] in done:
            continue
        meta = req["meta"]
        completions = []
        for _ in range(n_continuations):
            if rng.random() < 0.05:
                completions.append(
                    {"text": "still thinking, never closes", "finish_reason": "length"}
                )
                continue
            t = meta["threshold"]
            side_above = meta["cached_estimate"] > t
            if rng.random() < 0.10:
                side_above = not side_above
            scale = 1.0 + rng.random()  # uniform in (1, 2)
            value = t * scale if side_above else t / scale
            completions.append(
                {
                    "text": f"\n</think>\n\n{value:.0f}\n\nJustification: fabricated.",
                    "finish_reason": "stop",
                }
            )
        rec = {
            "req_id": req["req_id"],
            "kind": req["kind"],
            "meta": meta,
            "completions": completions,
            "usage": {},
        }
        asyncio.run(writer.write(rec))
    writer.close()


# --- estimate extraction --------------------------------------------------


def regex_estimate(answer: str) -> float | None:
    """First number in the answer text. Offline fallback instrument only."""
    if not answer:
        return None
    m = _NUMBER_RE.search(answer)
    if not m:
        return None
    s = m.group(0).replace(",", "").replace("$", "")
    m2 = re.match(r"(-?\d+(?:\.\d+)?)(?:\s*(?:x|×)\s*10\^?(\d+)|[eE]([+-]?\d+))?", s)
    if not m2:
        return None
    value = float(m2.group(1))
    exponent = m2.group(2) or m2.group(3)
    if exponent:
        value *= 10.0 ** int(exponent)
    return value


def _screen_judge_module():
    """Import screen/judge.py (paper's judge template + parser, our transport)."""
    screen_dir = Path(__file__).resolve().parent.parent / "screen"
    if str(screen_dir) not in sys.path:
        sys.path.insert(0, str(screen_dir))
    import judge as screen_judge

    return screen_judge


def extract_estimates(
    results_path: Path, *, use_regex: bool
) -> tuple[dict[str, list[float | None]], int]:
    """Map source_id -> per-continuation estimates (None = unparseable).

    A continuation that never closes `</think>` has an empty answer and is
    unparseable by construction. Tagged answers are honoured first (the
    parser is free and authoritative when present); everything else goes to
    one instrument — the screen's LLM judge by default, the regex under
    `--regex-judge`. Judge verdicts are cached in the results file keyed by
    `answer_cache_key`, so a re-run never re-judges identical answer text.
    """
    judged: dict[str, float | None] = {}
    for rec in iter_jsonl(results_path):
        if rec.get("kind") == "judge_cache":
            judged[rec["answer_key"]] = rec["estimate"]

    per_source: dict[str, list] = {}
    to_judge: list[tuple[str, str, str]] = []  # (answer_key, prompt_key, answer)
    for rec in iter_jsonl(results_path):
        if rec.get("error") or rec.get("kind") != "continue":
            continue
        meta = rec["meta"]
        slots = per_source.setdefault(meta["source_id"], [])
        for completion in rec["completions"]:
            _, answer = split_completion(completion["text"])
            if not answer.strip():
                slots.append(None)
                continue
            tagged = parse_tagged_estimate(answer)
            if tagged is not None:
                slots.append(tagged)
                continue
            if use_regex:
                slots.append(regex_estimate(answer))
                continue
            key = answer_cache_key(meta["prompt_key"], answer)
            if key not in judged:
                to_judge.append((key, meta["prompt_key"], answer))
            slots.append(("judge", key))

    if to_judge and not use_regex:
        screen_judge = _screen_judge_module()
        unique = {key: (pk, ans) for key, pk, ans in to_judge}
        keys = list(unique)
        print(f"judging {len(keys)} unique answers via screen/judge.py", flush=True)
        estimates = asyncio.run(
            screen_judge.judge_texts_async([unique[k] for k in keys])
        )
        writer = _Writer(results_path, fsync_every=1)
        for key, est in zip(keys, estimates):
            judged[key] = est
            asyncio.run(
                writer.write(
                    {"req_id": f"judge:{key}", "kind": "judge_cache",
                     "answer_key": key, "estimate": est}
                )
            )
        writer.close()

    n_unparsed = 0
    resolved: dict[str, list[float | None]] = {}
    for source_id, slots in per_source.items():
        row = []
        for slot in slots:
            est = judged.get(slot[1]) if isinstance(slot, tuple) else slot
            if est is None:
                n_unparsed += 1
            row.append(est)
        resolved[source_id] = row
    return resolved, n_unparsed


# --- statistics -----------------------------------------------------------


def binom_tail(k: int, n: int) -> float:
    """Exact one-sided P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n


def ks_statistic(a: list[float], b: list[float]) -> float | None:
    """Two-sample KS D statistic (descriptive only, no p-value)."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    d = 0.0
    ia = ib = 0
    values = sorted(set(a) | set(b))
    for v in values:
        while ia < len(a) and a[ia] <= v:
            ia += 1
        while ib < len(b) and b[ib] <= v:
            ib += 1
        d = max(d, abs(ia / len(a) - ib / len(b)))
    return d


def summarise(
    sources: pd.DataFrame,
    estimates: dict[str, list[float | None]],
    n_unparsed: int,
    *,
    judge_mode: str,
) -> dict:
    """Preregistered comparison: per-rollout side agreement + sign test."""
    per_rollout = []
    pooled_continuations: list[float] = []
    cached_estimates: list[float] = []
    n_agree = 0
    n_effective = 0
    for _, row in sources.iterrows():
        source_id = row["rollout_id"]
        threshold = float(row["threshold"])
        cached = float(row["estimate"])
        cached_side = cached > threshold
        ests = estimates.get(source_id, [])
        parsed = [e for e in ests if e is not None]
        pooled_continuations.extend(parsed)
        cached_estimates.append(cached)
        same = [(e > threshold) == cached_side for e in parsed]
        frac_same = sum(same) / len(parsed) if parsed else None
        # Ties (exactly half) count as disagreement — conservative.
        agree = frac_same is not None and frac_same > 0.5
        if parsed:
            n_effective += 1
            n_agree += int(agree)
        per_rollout.append(
            {
                "source_id": source_id,
                "prompt_key": row["prompt_key"],
                "direction": row["direction"],
                "threshold": threshold,
                "cached_estimate": cached,
                "cached_side_above": cached_side,
                # Near 1.0 = the cached estimate sits close to the threshold,
                # so side agreement is fragile under resampling even with a
                # perfect splice. Read failures through this column.
                "cached_over_threshold": cached / threshold if threshold else None,
                "n_parsed": len(parsed),
                "n_unparsed": len(ests) - len(parsed),
                "frac_same_side": frac_same,
                "majority_agrees": agree,
            }
        )
    binom_p = binom_tail(n_agree, n_effective)
    return {
        "n_sources": len(sources),
        "n_effective": n_effective,
        "majority_agree": n_agree,
        "binom_p": binom_p,
        "pass": bool(n_effective > 0 and binom_p < 0.01),
        "ks_stat": ks_statistic(pooled_continuations, cached_estimates),
        "n_unparsed": n_unparsed,
        "judge_mode": judge_mode,
        "per_rollout": per_rollout,
    }


# --- CLI ------------------------------------------------------------------


def resolve_model(base_url: str, model: str | None) -> str:
    """Use --model if given, else ask the server what it is serving."""
    if model:
        return model
    import httpx

    resp = httpx.get(base_url.rstrip("/") + "/models", timeout=30.0)
    resp.raise_for_status()
    served = [m["id"] for m in resp.json().get("data", [])]
    if not served:
        raise RuntimeError(f"{base_url}/models lists no models; pass --model")
    if len(served) > 1:
        print(f"multiple served models {served}, using {served[0]}", flush=True)
    return served[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-parquet", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None,
                        help="served model name; default: query {base-url}/models")
    parser.add_argument("--n-sources", type=int, default=20)
    parser.add_argument("--n-continuations", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None,
                        help="results jsonl (append-only, resumable); default "
                             "splice_results.jsonl, .dryrun variant under --dry-run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--regex-judge", action="store_true",
                        help="first-number regex instead of the LLM judge "
                             "(offline; 97.2% side-agreement with cached judge)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fabricate continuations, no server or API key")
    args = parser.parse_args(argv)

    out = args.out or Path(
        "splice_results.dryrun.jsonl" if args.dry_run else "splice_results.jsonl"
    )
    sources = sample_sources(args.master_parquet, args.n_sources, args.seed)
    model = DEFAULT_MODEL if args.dry_run else resolve_model(args.base_url, args.model)
    requests = build_splice_requests(
        sources, model=model, n_continuations=args.n_continuations
    )
    print(
        f"{len(requests)} sources x {args.n_continuations} continuations, "
        f"model {model}, results -> {out}",
        flush=True,
    )

    if args.dry_run:
        fabricate_results(
            requests, out, n_continuations=args.n_continuations, seed=args.seed
        )
        print(f"PROGRESS splice {len(requests)}/{len(requests)}", flush=True)
    else:
        asyncio.run(
            run_splice_requests(
                requests, out,
                base_url=args.base_url,
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
        )

    estimates, n_unparsed = extract_estimates(
        out, use_regex=args.regex_judge or args.dry_run
    )
    judge_mode = "regex" if (args.regex_judge or args.dry_run) else "llm"
    summary = summarise(sources, estimates, n_unparsed, judge_mode=judge_mode)
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
