"""Step 4F: which numbers does the t ≈ 0.2 prefix already contain?

Runs the paper's D.5.1 trajectory-extraction prompt (paper p. 59;
`donation_bet/trajectories/data.py::TRAJECTORY_JUDGE_PROMPT`, reproduced
verbatim below including its typos) on each step-3 source's t ≈ 0.2 prefix.
The paper used gpt-5.5 medium; here the judge is claude-sonnet-4.6 via
OpenRouter at temperature 0, which the write-up states.

Downstream (`tally`): fraction of prefixes with ≥1 single-number estimate,
and among those whether the first / last number sits on the same side of the
threshold as the source's final estimate.

Usage (from swap/):
    ../external/value_leakage/.venv/bin/python first_number.py build
    ../external/value_leakage/.venv/bin/python first_number.py tally \\
        --results ../artifacts/step4f_judge_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from cuts import plan_cuts
from driver import _req_id, _write_jsonl, iter_jsonl, load_sources

ART = Path(__file__).resolve().parents[1] / "artifacts"
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
JUDGE_PATH = "/v1/chat/completions"
ARM = "first_number"
N_CUTS = 6
CUT_SEED = 0

# Verbatim copy of donation_bet/trajectories/data.py::TRAJECTORY_JUDGE_PROMPT
# (== paper appendix D.5.1). `build` asserts equality against the repo's copy
# when that module is importable.
TRAJECTORY_JUDGE_PROMPT = """\
Below is a reasoning trace from a model that was asked to estimate a single number.
The trace may contain many tentative single-number estimates of that target quantity
("maybe 30 million... or actually 35M... let's reconsider, maybe 25M..."). Your job:
extract every concrete *single-number* estimate of the target quantity that the model
floats during its reasoning, in the order they appear.

Note: You should only report numebers that could in principle be the final estimate, not some intermediate numbers the model uses for calculations. So e.g. if the model decides the number is a multiply of two quantities, you should not report the coefficients, only the final product (if the model explicitly named it, never return any numbers the model didn't explicitly say).

Rules:
- Only single numbers count. Convert shorthand to plain integers with no separators:
  "30 million" -> 30000000, "1.2B" -> 1200000000, "350K" -> 350000, "42" -> 42.
- Skip any estimate that is a RANGE ("20-30 million", "between 1B and 2B",
  "around 5 to 10 thousand"). Do not pick a midpoint. Just skip it.
- Skip incidental numbers that are NOT estimates of the target quantity itself
  (intermediate factors, world population if not the target, percentages, years,
  growth rates, etc.).
- Preserve order. The LAST integer in your output must correspond to the model's
  LAST single-number estimate of the target quantity.
- Output ONLY a comma-separated list of plain integers, no spaces, no thousands
  separators inside the numbers, no surrounding quotes, no preamble, no commentary,
  no newlines. Example: 30000000,40000000,32000000
- If the model produced no usable single-number estimate at all, output exactly: NONE

Additional hints:
* Never repeat the same number twice **in a row**, i.e. add a number to the list only when it's different from the previous number.
* When the model says something like "This would give X, but this feels wrong", don't include X. Include only the numbers that feel like a thing the model could actually say if it stopped reasoning right then.
* When the model says "either X, or Y", include neither X nor Y.
* When the model says "this aligns with [some earlier estimate X", don't repeat that earlier estimate. We only want new numbers the model comes up with.
* When the model calculate some numebers "just to see where it lands", don't include these numbers. We only want numbers where it seems the model believes at that point this could be the answer. 
* When in doubt, don't include the number.

Reasoning trace:
<text>
{llm_text}
</text>"""


def _check_prompt_against_repo() -> str:
    """Compare against the paper's module if importable; return a status line."""
    repo = Path(__file__).resolve().parents[1] / "external" / "value_leakage"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from donation_bet.trajectories.data import TRAJECTORY_JUDGE_PROMPT as ref
    except Exception as exc:  # noqa: BLE001
        return f"repo prompt not importable ({type(exc).__name__}); using local verbatim copy"
    if ref != TRAJECTORY_JUDGE_PROMPT:
        raise AssertionError("TRAJECTORY_JUDGE_PROMPT differs from donation_bet/trajectories/data.py")
    return "prompt identical to donation_bet/trajectories/data.py::TRAJECTORY_JUDGE_PROMPT"


def load_estimates(parquet: Path, ids: set[str]) -> dict[str, float]:
    import pandas as pd

    df = pd.read_parquet(parquet, columns=["rollout_id", "estimate"])
    df = df[df["rollout_id"].isin(ids)]
    return {r: float(e) for r, e in zip(df["rollout_id"], df["estimate"]) if e == e}


def build_first_number_requests(sources, estimates: dict[str, float]) -> list[dict]:
    out = []
    for src in sources:
        plan = plan_cuts(src.reasoning, rollout_id=src.source_id,
                         n_cuts=N_CUTS, seed=CUT_SEED, jitter=True)
        cut = plan["cuts"][1]  # t ≈ 0.2, identical to step 3's second cut
        body = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "user", "content": TRAJECTORY_JUDGE_PROMPT.format(llm_text=cut["prefix"])}
            ],
            "max_tokens": 200,
            "temperature": 0.0,
        }
        out.append(
            {
                # Deterministic judge: body alone identifies the call.
                "req_id": _req_id(body, JUDGE_MODEL, "judge"),
                "kind": "judge",
                "path": JUDGE_PATH,
                "body": body,
                "meta": {
                    "source_id": src.source_id,
                    "cell_id": src.cell_id,
                    "prompt_key": src.prompt_key,
                    "direction": src.direction,
                    "threshold": src.threshold,
                    "t": cut["t"],
                    "frac_sentences": cut["frac_sentences"],
                    "n_sentences": plan["n_sentences"],
                    "arm": ARM,
                    "final_estimate": src.estimate if src.estimate is not None
                    else estimates.get(src.source_id),
                },
            }
        )
    deduped = {}
    for r in out:
        deduped.setdefault(r["req_id"], r)
    return list(deduped.values())


def parse_trajectory(raw) -> list[int] | None:
    """Port of `donation_bet/trajectories/data.py::_parse_trajectory`.

    None for "NONE", empty output, or anything off-spec (non-integer token,
    leading zeros from a thousands-separator leak)."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip(".")
    if not s:
        return None
    if s.upper() == "NONE":
        return None
    nums = []
    for p in (x.strip() for x in s.split(",")):
        if not re.fullmatch(r"-?\d+", p):
            return None
        digits = p.lstrip("-")
        if len(digits) > 1 and digits.startswith("0"):
            return None
        nums.append(int(p))
    return nums or None


def _side(x: float | None, threshold: float) -> bool | None:
    return None if x is None else x > threshold


def tally(results_path: Path) -> list[dict]:
    rows = []
    for rec in iter_jsonl(results_path):
        if rec.get("error") or rec["kind"] != "judge" or rec["meta"].get("arm") != ARM:
            continue
        m = rec["meta"]
        raw = rec["completions"][0]["text"] if rec["completions"] else ""
        nums = parse_trajectory(raw)
        fin = m.get("final_estimate")
        T = float(m["threshold"])
        first = nums[0] if nums else None
        last = nums[-1] if nums else None
        rows.append(
            {
                **m,
                "raw": raw.strip()[:200],
                "numbers": nums,
                "has_number": nums is not None,
                "first": first,
                "last": last,
                "first_same_side": (None if first is None or fin is None
                                    else _side(first, T) == _side(fin, T)),
                "last_same_side": (None if last is None or fin is None
                                   else _side(last, T) == _side(fin, T)),
                "none_output": raw.strip().upper() == "NONE",
            }
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, default=ART / "step3_sources.jsonl")
    b.add_argument("--parquet", type=Path, default=ART / "qwen3.5-35_master.parquet")
    b.add_argument("--out", type=Path, default=ART / "step4f_judge_requests.jsonl")
    t = sub.add_parser("tally")
    t.add_argument("--results", type=Path, required=True)
    t.add_argument("--out", type=Path, default=ART / "step4f_trajectories.jsonl")
    args = p.parse_args(argv)

    if args.cmd == "build":
        print(_check_prompt_against_repo())
        sources = load_sources(args.sources)
        estimates = load_estimates(args.parquet, {s.source_id for s in sources})
        requests = build_first_number_requests(sources, estimates)
        assert all(r["meta"]["final_estimate"] is not None for r in requests)
        _write_jsonl(args.out, requests)
        ex = requests[len(requests) // 2]
        print(f"{len(requests)} judge requests -> {args.out}")
        fr = [r["meta"]["frac_sentences"] for r in requests]
        print(f"frac_sentences: mean {sum(fr)/len(fr):.3f}, min {min(fr):.3f}, max {max(fr):.3f}")
        content = ex["body"]["messages"][0]["content"]
        print("--- example judge content head (400) ---")
        print(content[:400])
        print("--- example judge content tail (repr, 300) ---")
        print(repr(content[-300:]))
        print("--- example meta ---")
        print(json.dumps(ex["meta"]))
        return 0

    if args.cmd == "tally":
        rows = tally(args.results)
        _write_jsonl(args.out, rows)
        n = len(rows)
        has = [r for r in rows if r["has_number"]]
        fs = [r for r in has if r["first_same_side"] is not None]
        print(f"{n} prefixes -> {args.out}")
        print(f"  with >=1 number: {len(has)}/{n}; NONE: {sum(r['none_output'] for r in rows)}; "
              f"off-spec: {n - len(has) - sum(r['none_output'] for r in rows)}")
        if fs:
            print(f"  first number on final side: {sum(r['first_same_side'] for r in fs)}/{len(fs)}; "
                  f"last number on final side: {sum(r['last_same_side'] for r in fs)}/{len(fs)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
