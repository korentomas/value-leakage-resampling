"""Prefix-swap continuation driver: request building, async execution, scoring.

Pipeline
--------
    sources.jsonl                 one line per source rollout (from the
                                  paper's cached df)
        │  build-requests
        ▼
    requests.jsonl                one line per HTTP call, content-addressed
        │  run
        ▼
    results.jsonl                 append-only, resumable; one line per
                                  completed request
        │  build-judge-requests → run → results
        ▼
    counts.jsonl                  k/N per (source, cut, arm) for model.py

Every stage is idempotent. `req_id` is a hash of the exact request body plus
the model name, so re-running `build-requests` after changing a prompt
produces new ids rather than silently reusing stale samples, and `run`
skips ids already present in `results.jsonl`.

The measurement
---------------
For source rollout *i* sampled under condition *c* (``above_good`` or
``below_good``) on question *q* with threshold *T*, and for each cut *t*:

  * **orig arm** — resume rollout *i*'s prefix under *q*'s prompt for
    condition *c*;
  * **swap arm** — resume the *same* prefix under *q*'s prompt for the
    opposite condition, same *T*.

Both arms are scored with the same predicate: *i*'s condition-favored side,
i.e. ``estimate > T`` when *c* is ``above_good`` and ``estimate <= T`` when
*c* is ``below_good``. Then

    dep_i(t) = P(favored | orig) - P(favored | swap).

t = 0 is a special case
-----------------------
At ``t = 0`` the prefix is empty, so the request does not depend on *i* at
all: every rollout in a (question, direction, threshold) cell would issue
byte-identical requests. Worse (better, for the budget) — the two t=0 arms
are just fresh samples from the two conditions, which the paper already has
~100 of per cell. Writing out the algebra with ``p_d`` = the cached
``on_good_side`` rate in direction *d*:

    dep(0) = p_above - (1 - p_below) = p_above + p_below - 1

which is exactly `donation_bet/bias_metrics.balanced_bias_score`. So
dep(0) is free, identical for both directions of a question, and already
validated against the paper's published figure. `--include-t0` exists for a
sanity check against fresh samples; the default skips it and
`counts-from-cache` fills the t=0 row from the cached df instead.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from cuts import plan_cuts
from template import STOP_STRINGS, render_with_prefix, split_completion

DIRECTIONS = ("below_good", "above_good")
OPPOSITE = {"below_good": "above_good", "above_good": "below_good"}

_ESTIMATE_TAG_RE = re.compile(
    r"<final_estimate>(.*?)</final_estimate>", re.DOTALL | re.IGNORECASE
)


# --- source records -------------------------------------------------------


@dataclass
class Source:
    """One cached rollout to be resampled.

    `prompt` is the fully rendered user turn under the rollout's own
    condition; `counter_prompt` is the same question and threshold under the
    opposite condition. Supply `counter_prompt` in the JSONL, or let
    `load_sources` build it from the paper's `THRESHOLD_PROMPTS` when the
    external repo is importable.
    """

    source_id: str
    prompt_key: str
    direction: str
    threshold: float
    prompt: str
    reasoning: str
    counter_prompt: str = ""
    answer: str = ""
    estimate: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        return f"{self.prompt_key}|{self.direction}|{self.threshold:g}"


def _threshold_prompts():
    """Import the paper's prompt tables, or return None if unavailable."""
    here = Path(__file__).resolve().parent
    repo = here.parent / "external" / "value_leakage"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from shared.prompts.thresholds import THRESHOLD_PROMPTS

        return THRESHOLD_PROMPTS
    except Exception:  # noqa: BLE001 - optional dependency on the clone
        return None


def build_counter_prompt(prompt_key: str, direction: str, threshold) -> str:
    """Render the opposite-direction prompt for the same question/threshold."""
    table = _threshold_prompts()
    if table is None:
        raise RuntimeError(
            "external/value_leakage is not importable; put `counter_prompt` "
            "in sources.jsonl instead"
        )
    other = OPPOSITE[direction]
    template = table[prompt_key][f"{other}_template"]
    return template.format(threshold=_fmt_threshold(threshold))


def _fmt_threshold(threshold) -> str:
    """Format a threshold the way the paper's runner does (integer-valued)."""
    # The paper's runner renders ``f"{t:,}"`` (shared/runner.py), i.e. with
    # thousands separators. Step 3 (2026-08-18) rendered the swap arm WITHOUT
    # separators for 7/9 questions (turns=950, zills=800 unaffected); see
    # FINDINGS "threshold-format mismatch". Fixed 2026-08-21 for anything new.
    f = float(threshold)
    return f"{int(f):,}" if f.is_integer() else f"{f:,}"


def load_sources(path: Path) -> list[Source]:
    sources = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = {
                "source_id",
                "prompt_key",
                "direction",
                "threshold",
                "prompt",
                "reasoning",
            } - rec.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields {sorted(missing)}")
            if rec["direction"] not in DIRECTIONS + ("baseline",):
                raise ValueError(f"{path}:{line_no} bad direction {rec['direction']!r}")
            known = {
                "source_id", "prompt_key", "direction", "threshold", "prompt",
                "reasoning", "counter_prompt", "answer", "estimate",
            }
            src = Source(
                source_id=str(rec["source_id"]),
                prompt_key=rec["prompt_key"],
                direction=rec["direction"],
                threshold=float(rec["threshold"]),
                prompt=rec["prompt"],
                reasoning=rec["reasoning"] or "",
                counter_prompt=rec.get("counter_prompt", ""),
                answer=rec.get("answer", ""),
                estimate=rec.get("estimate"),
                extra={k: v for k, v in rec.items() if k not in known},
            )
            if not src.counter_prompt and src.direction in DIRECTIONS:
                # Baseline sources (step 4D) have no opposite condition.
                src.counter_prompt = build_counter_prompt(
                    src.prompt_key, src.direction, src.threshold
                )
            sources.append(src)
    return sources


# --- request construction -------------------------------------------------


def _req_id(body: dict, model: str, kind: str, discriminator=None) -> str:
    """Content-addressed request id.

    The body alone is not enough to identify a continuation request: the
    chunks that make up one (source, cut, arm) draw have byte-identical
    bodies when no explicit sampling seed is set, so hashing the body alone
    would collapse all of them into one request and silently sample 20x
    fewer continuations than asked for. The discriminator — which chunk of
    which arm of which source — restores the distinction while keeping the
    id stable across rebuilds.
    """
    payload = json.dumps(
        {"model": model, "kind": kind, "body": body, "d": discriminator},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _continuation_body(
    prompt: str, *, model: str, n: int, max_tokens: int, temperature: float, seed: int | None
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stop": STOP_STRINGS,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def build_requests(
    sources: list[Source],
    *,
    model: str,
    n_cuts: int = 6,
    n_continuations: int = 20,
    chunk: int = 20,
    max_tokens: int = 16000,
    temperature: float = 1.0,
    include_t0: bool = False,
    n_t0: int = 100,
    cut_seed: int = 0,
    jitter: bool = True,
    sample_seed: int | None = None,
    token_budget: int | None = 16000,
    count_tokens=None,
) -> list[dict]:
    """Build every continuation request for a batch of sources.

    `chunk` splits the `n_continuations` samples across several requests with
    `n=chunk` each, so an interrupted run loses at most one chunk. vLLM
    prefill-caches the shared prefix, so chunking costs little.

    With `include_t0`, the t=0 requests are emitted once per
    (question, direction, threshold) cell rather than once per rollout, with
    `n_t0` samples per arm.

    `token_budget` matches the paper's total sampling budget (their
    `max_tokens=16000` in `shared/models.py`). On the completions endpoint
    `max_tokens` counts *new* tokens, so a continuation from a long prefix
    would otherwise be allowed to run past the budget its source rollout was
    sampled under, making late cuts systematically less likely to hit the
    cap than early ones — a cut-position artefact in exactly the quantity we
    are measuring. Each request's cap is therefore reduced by the prefix
    length. Pass `count_tokens` (e.g. `tokenizer.encode` composed with
    `len`) for an exact count; the default is a 4-characters-per-token
    approximation, which is close enough for a cap but should be replaced
    with the real tokenizer on the serving box.
    """
    if count_tokens is None:
        def count_tokens(text: str) -> int:
            return len(text) // 4
    requests: list[dict] = []
    seen_t0_cells: set[str] = set()

    for src in sources:
        plan = plan_cuts(
            src.reasoning,
            rollout_id=src.source_id,
            n_cuts=n_cuts,
            seed=cut_seed,
            jitter=jitter,
        )
        for cut in plan["cuts"]:
            t = cut["t"]
            is_t0 = t == 0
            if is_t0:
                if not include_t0:
                    continue
                if src.cell_id in seen_t0_cells:
                    continue
            prefix_tokens = count_tokens(cut["prefix"])
            if token_budget is None:
                cut_max_tokens = max_tokens
            else:
                cut_max_tokens = max(256, min(max_tokens, token_budget - prefix_tokens))
            for arm, user in (("orig", src.prompt), ("swap", src.counter_prompt)):
                prompt = render_with_prefix(user, cut["prefix"])
                total = n_t0 if is_t0 else n_continuations
                for offset in range(0, total, chunk):
                    k = min(chunk, total - offset)
                    unit = src.cell_id if is_t0 else src.source_id
                    discriminator = [unit, t, arm, offset]
                    seed = None
                    if sample_seed is not None:
                        # Distinct per request, so two chunks (or two
                        # rollouts that happen to share a prefix) never
                        # replay the same samples.
                        seed = (
                            sample_seed
                            + int(
                                hashlib.blake2b(
                                    json.dumps(discriminator).encode(), digest_size=4
                                ).hexdigest(),
                                16,
                            )
                        ) % (2**31 - 1)
                    body = _continuation_body(
                        prompt,
                        model=model,
                        n=k,
                        max_tokens=cut_max_tokens,
                        temperature=temperature,
                        seed=seed,
                    )
                    requests.append(
                        {
                            "req_id": _req_id(body, model, "continue", discriminator),
                            "kind": "continue",
                            "path": "/v1/completions",
                            "body": body,
                            "meta": {
                                "source_id": None if is_t0 else src.source_id,
                                "cell_id": src.cell_id,
                                "prompt_key": src.prompt_key,
                                "direction": src.direction,
                                "threshold": src.threshold,
                                "arm": arm,
                                "t": t,
                                "char_offset": cut["char_offset"],
                                "n_sentences": plan["n_sentences"],
                                "frac_sentences": cut["frac_sentences"],
                                "prefix_tokens_est": prefix_tokens,
                                "chunk_offset": offset,
                            },
                        }
                    )
            if is_t0:
                seen_t0_cells.add(src.cell_id)

    # Content-addressed ids collapse duplicates (identical prefix text across
    # rollouts does happen at small t).
    deduped = {}
    for req in requests:
        deduped.setdefault(req["req_id"], req)
    return list(deduped.values())


# --- judge requests -------------------------------------------------------


def build_judge_requests(
    results_path: Path,
    *,
    judge_model: str = "gpt-4.1",
    judge_path: str = "/v1/chat/completions",
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> list[dict]:
    """Turn continuation answers into estimate-judge requests.

    Reuses the paper's per-question judge template so our estimates are
    produced by the same instrument as theirs. Answers that parse as a bare
    number are still sent: mixing a regex path with a judge path would make
    our estimates a different instrument from theirs, and the judge is a
    rounding error in the budget (see README).
    """
    table = _threshold_prompts()
    if table is None:
        raise RuntimeError("external/value_leakage is not importable")

    requests = []
    for rec in iter_jsonl(results_path):
        if rec.get("error") or rec["kind"] != "continue":
            continue
        meta = rec["meta"]
        template = table[meta["prompt_key"]]["judge_prompt"]
        for idx, completion in enumerate(rec["completions"]):
            _, answer = split_completion(completion["text"])
            answer_key = answer_cache_key(meta["prompt_key"], answer)
            if not answer.strip():
                # Never closed </think>: unparseable by construction, and
                # sending an empty body to the judge only wastes tokens.
                continue
            body = {
                "model": judge_model,
                "messages": [
                    {"role": "user", "content": template.format(llm_text=answer)}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            requests.append(
                {
                    # The judge is deterministic given its input, so two
                    # continuations with identical answer text legitimately
                    # share one judge call — the body alone is the right key.
                    "req_id": _req_id(body, judge_model, "judge"),
                    "kind": "judge",
                    "path": judge_path,
                    "body": body,
                    "meta": {
                        # Estimates are looked up by `answer_key`, not by
                        # parent: identical answer text is judged once and
                        # the result is shared by every continuation that
                        # produced it. The parent fields below name the
                        # first such continuation, for traceability only.
                        "answer_key": answer_key,
                        "parent_req_id": rec["req_id"],
                        "completion_index": idx,
                        **{
                            k: meta[k]
                            for k in (
                                "source_id", "cell_id", "prompt_key",
                                "direction", "threshold", "arm", "t",
                            )
                        },
                    },
                }
            )
    deduped = {}
    for req in requests:
        deduped.setdefault(req["req_id"], req)
    return list(deduped.values())


def answer_cache_key(prompt_key: str, answer: str) -> str:
    """Identity of a judge call: the question's template plus the answer text."""
    h = hashlib.blake2b(digest_size=16)
    h.update(prompt_key.encode())
    h.update(b"\x00")
    h.update(answer.encode())
    return h.hexdigest()


def parse_tagged_estimate(raw) -> float | None:
    """Port of `shared/runner.py::_parse_tagged_estimate`."""
    if not isinstance(raw, str):
        return None
    m = _ESTIMATE_TAG_RE.search(raw)
    if not m:
        return None
    content = m.group(1).strip()
    if not content or content.upper() == "UNKNOWN":
        return None
    try:
        return float(content.replace(",", ""))
    except ValueError:
        return None


# --- resumable async runner ----------------------------------------------


def iter_jsonl(path: Path):
    if not Path(path).exists():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave one truncated final line.
                continue


def completed_ids(path: Path) -> set[str]:
    """req_ids already recorded, excluding retryable failures."""
    done = set()
    for rec in iter_jsonl(path):
        if rec.get("error") and rec.get("retryable", False):
            continue
        done.add(rec["req_id"])
    return done


class _Writer:
    """Serialises appends so partial lines cannot interleave."""

    def __init__(self, path: Path, fsync_every: int = 50):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._lock = asyncio.Lock()
        self._since_sync = 0
        self._fsync_every = fsync_every

    async def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._since_sync += 1
            if self._since_sync >= self._fsync_every:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._since_sync = 0

    def close(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()


def _extract_completions(kind: str, payload: dict) -> list[dict]:
    """Normalise a completions or chat-completions response."""
    out = []
    for choice in payload.get("choices", []):
        if kind == "judge":
            text = (choice.get("message") or {}).get("content", "")
        else:
            text = choice.get("text", "")
        out.append({"text": text, "finish_reason": choice.get("finish_reason")})
    return out


async def _post_one(client, base_url: str, req: dict, *, timeout: float, max_retries: int):
    """POST one request with backoff. Returns a result record."""
    url = base_url.rstrip("/") + req["path"]
    delay = 2.0
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, json=req["body"], timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                return {
                    "req_id": req["req_id"],
                    "kind": req["kind"],
                    "meta": req["meta"],
                    "completions": _extract_completions(req["kind"], payload),
                    "usage": payload.get("usage", {}),
                }
            body = resp.text[:400]
            if resp.status_code in (400, 401, 403, 404, 422):
                return {
                    "req_id": req["req_id"],
                    "kind": req["kind"],
                    "meta": req["meta"],
                    "error": f"HTTP {resp.status_code}: {body}",
                    "retryable": False,
                }
            last_err = f"HTTP {resp.status_code}: {body}"
        except Exception as exc:  # noqa: BLE001 - network layer, retry all
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            await asyncio.sleep(delay * (1 + random.random()))
            delay = min(delay * 2, 60.0)
    return {
        "req_id": req["req_id"],
        "kind": req["kind"],
        "meta": req["meta"],
        "error": last_err,
        "retryable": True,
    }


async def run_requests(
    requests: list[dict],
    results_path: Path,
    *,
    base_url: str,
    api_key: str | None = None,
    concurrency: int = 64,
    timeout: float = 1800.0,
    max_retries: int = 5,
    progress_every: float = 30.0,
) -> dict:
    """Execute pending requests, appending results as they land.

    Already-recorded `req_id`s are skipped, so re-running after any kind of
    interruption resumes exactly where it stopped.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - deployment-time
        raise RuntimeError("run_requests needs httpx: pip install httpx") from exc

    done = completed_ids(results_path)
    pending = [r for r in requests if r["req_id"] not in done]
    stats = {
        "total": len(requests),
        "already_done": len(requests) - len(pending),
        "attempted": len(pending),
        "ok": 0,
        "failed": 0,
        "completion_tokens": 0,
    }
    if not pending:
        return stats

    writer = _Writer(results_path)
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    start = time.time()
    last_report = start

    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency
    )) as client:

        async def worker(req):
            nonlocal last_report
            async with sem:
                rec = await _post_one(
                    client, base_url, req, timeout=timeout, max_retries=max_retries
                )
            await writer.write(rec)
            if rec.get("error"):
                stats["failed"] += 1
            else:
                stats["ok"] += 1
                stats["completion_tokens"] += rec.get("usage", {}).get(
                    "completion_tokens", 0
                )
            now = time.time()
            if now - last_report >= progress_every:
                last_report = now
                elapsed = now - start
                finished = stats["ok"] + stats["failed"]
                rate = finished / elapsed if elapsed else 0.0
                remaining = (len(pending) - finished) / rate if rate else float("nan")
                print(
                    f"  {finished}/{len(pending)} done, {stats['failed']} failed, "
                    f"{stats['completion_tokens'] / max(elapsed, 1e-9):,.0f} out-tok/s, "
                    f"eta {remaining / 60:.1f} min",
                    flush=True,
                )

        try:
            await asyncio.gather(*(worker(r) for r in pending))
        finally:
            writer.close()
    return stats


# --- scoring / aggregation ------------------------------------------------


def favored_side(estimate: float | None, direction: str, threshold: float) -> bool | None:
    """Is `estimate` on the side that the source rollout's condition favours?

    Both arms use the *source's* condition, which is what makes dep a signed
    quantity in a fixed direction rather than two unrelated rates.
    """
    if estimate is None:
        return None
    if direction == "above_good":
        return estimate > threshold
    if direction == "below_good":
        return estimate <= threshold
    if direction == "baseline":
        # Step 4D null arm: no favoured side exists, so score "above" and let
        # the analysis fold to |k/n - 0.5| (baseline_null.py).
        return estimate > threshold
    raise ValueError(f"bad direction {direction!r}")


def aggregate_counts(results_path: Path, judge_results_path: Path) -> list[dict]:
    """Collapse continuation + judge results into per-(source, t, arm) counts.

    Returns records with `k` (favoured-side count) and `n` (parsed count).
    Continuations whose estimate the judge could not parse are dropped from
    `n`, matching the paper's parse filtering in `_add_good_side`.
    """
    estimates: dict[str, float | None] = {}
    for rec in iter_jsonl(judge_results_path):
        if rec.get("error") or rec["kind"] != "judge" or not rec["completions"]:
            continue
        estimates[rec["meta"]["answer_key"]] = parse_tagged_estimate(
            rec["completions"][0]["text"]
        )

    cells: dict[tuple, dict] = {}
    for rec in iter_jsonl(results_path):
        if rec.get("error") or rec["kind"] != "continue":
            continue
        meta = rec["meta"]
        key = (
            meta["source_id"] or meta["cell_id"],
            meta["cell_id"],
            meta["t"],
            meta["arm"],
        )
        cell = cells.setdefault(
            key,
            {
                "unit_id": key[0],
                "is_cell_level": meta["source_id"] is None,
                "cell_id": meta["cell_id"],
                "prompt_key": meta["prompt_key"],
                "direction": meta["direction"],
                "threshold": meta["threshold"],
                "t": meta["t"],
                "arm": meta["arm"],
                "frac_sentences": meta.get("frac_sentences", 0.0),
                "n_sentences": meta.get("n_sentences", 0),
                "k": 0,
                "n": 0,
                "n_unparsed": 0,
                "n_truncated": 0,
            },
        )
        for completion in rec["completions"]:
            if completion.get("finish_reason") == "length":
                cell["n_truncated"] += 1
            _, answer = split_completion(completion["text"])
            est = estimates.get(answer_cache_key(meta["prompt_key"], answer))
            side = favored_side(est, meta["direction"], meta["threshold"])
            if side is None:
                cell["n_unparsed"] += 1
                continue
            cell["n"] += 1
            cell["k"] += int(side)
    return list(cells.values())


def counts_from_cache(df, *, prompt_keys=None) -> list[dict]:
    """t=0 counts straight from the paper's cached dataframe.

    dep(0) does not depend on the rollout, so a single (question, direction,
    threshold) cell supplies both arms:

      * orig arm = the cell's own rollouts scored on the favoured side,
      * swap arm = the opposite direction's rollouts scored on the *same*
        favoured side.

    `df` is the frame returned by `shared.get_main_dfs.get_main_dfs`.
    """
    out = []
    keys = prompt_keys if prompt_keys is not None else sorted(df["prompt_key"].unique())
    for prompt_key in keys:
        sub = df[df["prompt_key"] == prompt_key]
        for threshold in sorted(sub["threshold"].unique()):
            cell = sub[sub["threshold"] == threshold]
            present = {d: cell[cell["direction"] == d] for d in DIRECTIONS}
            if any(len(v) == 0 for v in present.values()):
                continue
            for direction in DIRECTIONS:
                cell_id = f"{prompt_key}|{direction}|{float(threshold):g}"
                for arm, source_dir in (
                    ("orig", direction),
                    ("swap", OPPOSITE[direction]),
                ):
                    rows = present[source_dir]
                    sides = [
                        favored_side(float(e), direction, float(threshold))
                        for e in rows["estimate"]
                    ]
                    sides = [s for s in sides if s is not None]
                    out.append(
                        {
                            "unit_id": cell_id,
                            "is_cell_level": True,
                            "cell_id": cell_id,
                            "prompt_key": prompt_key,
                            "direction": direction,
                            "threshold": float(threshold),
                            "t": 0,
                            "arm": arm,
                            "frac_sentences": 0.0,
                            "n_sentences": 0,
                            "k": int(sum(sides)),
                            "n": len(sides),
                            "n_unparsed": 0,
                            "n_truncated": 0,
                        }
                    )
    return out


def baseline_counts_from_cache(df, *, prompt_keys=None, arm: str = "baseline") -> list[dict]:
    """Per-cell counts of the paper's *no-bet* rollouts on each direction's
    favoured side (step 4A, estimand ``s = p_neutral - p_base``).

    The cached frame holds three kinds of rows per question: the two
    directional conditions (``direction`` in ``below_good`` / ``above_good``,
    with a threshold) and the baseline prompt (``direction == "baseline"``,
    threshold NaN). Thresholds are the baseline medians, so the rate comes
    out near 0.5 by construction; it is still estimated from the rows rather
    than assumed, because ties (``estimate == threshold``) and parse drops
    move it.

    One record per (question, direction, threshold) cell, ``unit_id =
    cell_id``, ``t = 0`` — the same shape as `counts_from_cache`, with
    ``arm`` set to `arm`. Called twice by `three_arm.py`: once as
    ``arm="baseline"`` (the constant arm B for s) and once as
    ``arm="neutral"`` (the free t=0 cell of the neutral arm, since an empty
    prefix under the baseline prompt *is* a baseline rollout).

    Assumption: p_base is a property of the question, shared by every unit
    in a (prompt_key, direction) cell.
    """
    out = []
    keys = prompt_keys if prompt_keys is not None else sorted(df["prompt_key"].unique())
    for prompt_key in keys:
        sub = df[df["prompt_key"] == prompt_key]
        base = sub[sub["direction"] == "baseline"]
        directional = sub[sub["direction"].isin(DIRECTIONS)]
        if len(base) == 0 or len(directional) == 0:
            continue
        estimates = [float(e) for e in base["estimate"] if e is not None and e == e]
        for threshold in sorted(directional["threshold"].dropna().unique()):
            for direction in DIRECTIONS:
                cell_id = f"{prompt_key}|{direction}|{float(threshold):g}"
                sides = [
                    favored_side(e, direction, float(threshold)) for e in estimates
                ]
                sides = [s for s in sides if s is not None]
                out.append(
                    {
                        "unit_id": cell_id,
                        "is_cell_level": True,
                        "cell_id": cell_id,
                        "prompt_key": prompt_key,
                        "direction": direction,
                        "threshold": float(threshold),
                        "t": 0,
                        "arm": arm,
                        "frac_sentences": 0.0,
                        "n_sentences": 0,
                        "k": int(sum(sides)),
                        "n": len(sides),
                        "n_unparsed": 0,
                        "n_truncated": 0,
                    }
                )
    return out


# --- CLI ------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _summarise_requests(requests: list[dict]) -> str:
    by_t = defaultdict(int)
    samples = 0
    prefix_chars = 0
    for req in requests:
        by_t[req["meta"]["t"]] += 1
        samples += req["body"].get("n", 1)
        prefix_chars += len(req["body"]["prompt"])
    lines = [
        f"{len(requests):,} requests, {samples:,} continuations",
        f"mean prompt length {prefix_chars / max(len(requests), 1):,.0f} chars",
        "requests per cut index: "
        + ", ".join(f"t={t}:{c}" for t, c in sorted(by_t.items())),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-requests", help="sources.jsonl -> requests.jsonl")
    p_build.add_argument("--sources", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--model", required=True)
    p_build.add_argument("--n-cuts", type=int, default=6)
    p_build.add_argument("--n-continuations", type=int, default=20)
    p_build.add_argument("--chunk", type=int, default=20)
    p_build.add_argument("--max-tokens", type=int, default=16000)
    p_build.add_argument("--temperature", type=float, default=1.0)
    p_build.add_argument("--include-t0", action="store_true")
    p_build.add_argument("--n-t0", type=int, default=100)
    p_build.add_argument("--cut-seed", type=int, default=0)
    p_build.add_argument("--no-jitter", action="store_true")
    p_build.add_argument("--sample-seed", type=int, default=None)
    p_build.add_argument(
        "--token-budget", type=int, default=16000,
        help="total sampling budget to match (0 disables the prefix subtraction)",
    )
    p_build.add_argument("--print-example", action="store_true")

    p_judge = sub.add_parser("build-judge-requests", help="results -> judge requests")
    p_judge.add_argument("--results", type=Path, required=True)
    p_judge.add_argument("--out", type=Path, required=True)
    p_judge.add_argument("--judge-model", default="gpt-4.1")
    p_judge.add_argument("--judge-path", default="/v1/chat/completions")

    p_run = sub.add_parser("run", help="execute requests, resumable")
    p_run.add_argument("--requests", type=Path, required=True)
    p_run.add_argument("--results", type=Path, required=True)
    p_run.add_argument("--base-url", required=True)
    p_run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p_run.add_argument("--concurrency", type=int, default=64)
    p_run.add_argument("--timeout", type=float, default=1800.0)
    p_run.add_argument("--max-retries", type=int, default=5)

    p_agg = sub.add_parser("aggregate", help="results -> counts.jsonl for model.py")
    p_agg.add_argument("--results", type=Path, required=True)
    p_agg.add_argument("--judge-results", type=Path, required=True)
    p_agg.add_argument("--out", type=Path, required=True)

    p_cache = sub.add_parser(
        "counts-from-cache",
        help="master parquet -> t=0 orig/swap cells and/or baseline cells (needs pandas+pyarrow)",
    )
    p_cache.add_argument("--parquet", type=Path, required=True)
    p_cache.add_argument("--out", type=Path, default=None, help="t=0 orig/swap cell counts")
    p_cache.add_argument(
        "--baseline-out", type=Path, default=None,
        help="per-cell baseline (no-bet) favoured-side counts, arm=baseline",
    )

    args = parser.parse_args(argv)

    if args.cmd == "build-requests":
        sources = load_sources(args.sources)
        requests = build_requests(
            sources,
            model=args.model,
            n_cuts=args.n_cuts,
            n_continuations=args.n_continuations,
            chunk=args.chunk,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            include_t0=args.include_t0,
            n_t0=args.n_t0,
            cut_seed=args.cut_seed,
            jitter=not args.no_jitter,
            sample_seed=args.sample_seed,
            token_budget=args.token_budget or None,
        )
        _write_jsonl(args.out, requests)
        print(f"{len(sources):,} sources -> {args.out}")
        print(_summarise_requests(requests))
        if args.print_example and requests:
            print("\n--- example prompt (repr) ---")
            print(repr(requests[len(requests) // 2]["body"]["prompt"]))
        return 0

    if args.cmd == "build-judge-requests":
        requests = build_judge_requests(
            args.results, judge_model=args.judge_model, judge_path=args.judge_path
        )
        _write_jsonl(args.out, requests)
        print(f"{len(requests):,} judge requests -> {args.out}")
        return 0

    if args.cmd == "run":
        requests = list(iter_jsonl(args.requests))
        api_key = os.environ.get(args.api_key_env)
        stats = asyncio.run(
            run_requests(
                requests,
                args.results,
                base_url=args.base_url,
                api_key=api_key,
                concurrency=args.concurrency,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
        )
        print(json.dumps(stats, indent=2))
        return 1 if stats["failed"] else 0

    if args.cmd == "aggregate":
        counts = aggregate_counts(args.results, args.judge_results)
        _write_jsonl(args.out, counts)
        total_n = sum(c["n"] for c in counts)
        total_unparsed = sum(c["n_unparsed"] for c in counts)
        total_trunc = sum(c["n_truncated"] for c in counts)
        print(f"{len(counts):,} count cells -> {args.out}")
        print(
            f"  {total_n:,} scored continuations, {total_unparsed:,} unparsed "
            f"({total_unparsed / max(total_n + total_unparsed, 1):.1%}), "
            f"{total_trunc:,} hit the token cap"
        )
        return 0

    if args.cmd == "counts-from-cache":
        import pandas as pd  # local: the model env does not carry pyarrow

        df = pd.read_parquet(args.parquet)
        if args.out is None and args.baseline_out is None:
            parser.error("give --out and/or --baseline-out")
        if args.out:
            counts = counts_from_cache(df[df["direction"].isin(DIRECTIONS)])
            _write_jsonl(args.out, counts)
            print(f"{len(counts)} t=0 orig/swap cells -> {args.out}")
        if args.baseline_out:
            counts = baseline_counts_from_cache(df)
            _write_jsonl(args.baseline_out, counts)
            rates = ", ".join(
                f"{c['cell_id']}: {c['k']}/{c['n']}" for c in counts
            )
            print(f"{len(counts)} baseline cells -> {args.baseline_out}\n  {rates}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
