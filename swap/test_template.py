"""Tests for chat-template rendering, completion splitting, and request building.

These check the *structure* our code assumes. The claim that the structure
matches the real Qwen3.5 checkpoint can only be settled by
``python template.py --verify`` on a box with the tokenizer, which this file
attempts and reports as SKIP when the dependency is missing.

Run: ``python test_template.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from cuts import plan_cuts
from driver import (
    Source,
    aggregate_counts,
    answer_cache_key,
    build_requests,
    favored_side,
    parse_tagged_estimate,
)
from template import (
    IM_END,
    IM_START,
    THINK_CLOSE,
    THINK_OPEN,
    reconstruct_assistant_text,
    render_generation_prompt,
    render_with_prefix,
    split_completion,
    verify_against_hf,
    verify_against_tinker,
)

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}")
        _FAILURES.append(label)


USER = "Estimate the number of giraffe spots. The threshold is 1000."


def test_prompt_shape():
    print("prompt shape")
    p = render_generation_prompt(USER)
    check("exact expected string", p == (
        f"{IM_START}user\n{USER}{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n"
    ), repr(p))
    check("no system turn is injected", "system" not in p, repr(p))
    check("user content is verbatim", USER in p)
    check("ends inside the think block", p.endswith(f"{THINK_OPEN}\n"), repr(p[-20:]))
    check("no newline between content and im_end", f"{USER}{IM_END}" in p)
    check("single newline before the assistant header",
          f"{IM_END}\n{IM_START}assistant" in p)


def test_prefix_append():
    print("prefix appending")
    prefix = "Let me start. Giraffes number about 117,000."
    p = render_with_prefix(USER, prefix)
    check("prefix lands directly after <think>\\n",
          p.endswith(f"{THINK_OPEN}\n{prefix}"), repr(p[-60:]))
    check("empty prefix equals the plain generation prompt",
          render_with_prefix(USER, "") == render_generation_prompt(USER))
    check("prompt does not end in a space", not p.endswith(" "))
    check("only one think-open tag", p.count(THINK_OPEN) == 1)
    check("no think-close tag", THINK_CLOSE not in p)


def test_split_completion():
    print("completion splitting")
    r, a = split_completion(
        " and so about 2.3e7.\n</think>\n\nMy best guess is 23,000,000." + IM_END
    )
    check("reasoning tail stripped", r == "and so about 2.3e7.", repr(r))
    check("answer has the separator newlines removed",
          a == "My best guess is 23,000,000.", repr(a))
    r2, a2 = split_completion("still reasoning, never closed the block")
    check("unterminated block yields an empty answer", a2 == "", repr(a2))
    check("unterminated block keeps the reasoning", r2.startswith("still reasoning"))

    full = reconstruct_assistant_text("thinking text", "the answer")
    r3, a3 = split_completion(full[len(THINK_OPEN) + 1:])
    check("round-trips through reconstruct_assistant_text",
          (r3, a3) == ("thinking text", "the answer"), repr((r3, a3)))


def test_estimate_parsing():
    print("estimate parsing")
    cases = [
        ("<final_estimate>2340000</final_estimate>", 2340000.0),
        ("<final_estimate>2,340,000</final_estimate>", 2340000.0),
        ("<final_estimate>UNKNOWN</final_estimate>", None),
        ("<final_estimate></final_estimate>", None),
        ("no tags at all", None),
        ("<final_estimate>1e7</final_estimate>", 1e7),
        (None, None),
    ]
    for raw, want in cases:
        got = parse_tagged_estimate(raw)
        check(f"parse {raw!r} -> {want}", got == want, f"got {got}")


def test_favored_side():
    print("favoured-side scoring")
    check("above_good favours exceeding the threshold",
          favored_side(1500.0, "above_good", 1000.0) is True)
    check("above_good: equality is not exceeding",
          favored_side(1000.0, "above_good", 1000.0) is False)
    check("below_good favours not exceeding",
          favored_side(1000.0, "below_good", 1000.0) is True)
    check("below_good: exceeding is unfavoured",
          favored_side(1500.0, "below_good", 1000.0) is False)
    check("unparsed estimate is None", favored_side(None, "below_good", 1.0) is None)


def _source(i: int, direction: str) -> Source:
    return Source(
        source_id=f"s{i}",
        prompt_key="v1_giraffes",
        direction=direction,
        threshold=1000.0,
        prompt=f"ORIGINAL PROMPT {direction} threshold 1000",
        counter_prompt=f"COUNTER PROMPT {direction} threshold 1000",
        reasoning=" ".join(f"Reasoning sentence {j}." for j in range(10)),
    )


def test_build_requests():
    print("request building")
    sources = [_source(0, "above_good"), _source(1, "below_good")]
    reqs = build_requests(
        sources, model="qwen", n_cuts=6, n_continuations=20, chunk=20
    )
    check("t=0 is skipped by default",
          all(r["meta"]["t"] != 0 for r in reqs), "found a t=0 request")
    # 2 sources x 5 paid cuts x 2 arms
    check("2 x 5 x 2 = 20 requests", len(reqs) == 20, str(len(reqs)))
    check("both arms present",
          {r["meta"]["arm"] for r in reqs} == {"orig", "swap"})
    check("req_ids are unique",
          len({r["req_id"] for r in reqs}) == len(reqs))
    check("orig arm uses the original prompt",
          all("ORIGINAL PROMPT" in r["body"]["prompt"]
              for r in reqs if r["meta"]["arm"] == "orig"))
    check("swap arm uses the counter prompt",
          all("COUNTER PROMPT" in r["body"]["prompt"]
              for r in reqs if r["meta"]["arm"] == "swap"))

    # The two arms of one (source, cut) must share the prefix exactly.
    by_key = {}
    for r in reqs:
        m = r["meta"]
        by_key.setdefault((m["source_id"], m["t"]), {})[m["arm"]] = r
    for (sid, t), arms in by_key.items():
        pre_o = arms["orig"]["body"]["prompt"].split(f"{THINK_OPEN}\n", 1)[1]
        pre_s = arms["swap"]["body"]["prompt"].split(f"{THINK_OPEN}\n", 1)[1]
        if pre_o != pre_s:
            check(f"arms share the prefix at ({sid}, t={t})", False,
                  f"{pre_o!r} vs {pre_s!r}")
            break
    else:
        check("both arms of every cut share the prefix exactly", True)

    check("stop sequence is set", all(r["body"]["stop"] == [IM_END] for r in reqs))
    check("temperature 1 by default",
          all(r["body"]["temperature"] == 1.0 for r in reqs))

    with_t0 = build_requests(
        sources, model="qwen", n_cuts=6, n_continuations=20, chunk=20,
        include_t0=True, n_t0=100,
    )
    t0 = [r for r in with_t0 if r["meta"]["t"] == 0]
    # Two sources sit in different cells (different directions): 2 cells
    # x 2 arms x (100 samples / chunk of 20) = 20 requests, and crucially
    # they are keyed by cell, not by rollout.
    check("t=0 is emitted per cell, not per rollout", len(t0) == 20, str(len(t0)))
    check("t=0 covers exactly the two cells",
          len({r["meta"]["cell_id"] for r in t0}) == 2)
    check("t=0 asks for n_t0 samples per cell and arm",
          sum(r["body"]["n"] for r in t0) == 2 * 2 * 100,
          str(sum(r["body"]["n"] for r in t0)))
    check("t=0 requests carry no source_id",
          all(r["meta"]["source_id"] is None for r in t0))

    chunked = build_requests(
        sources, model="qwen", n_cuts=6, n_continuations=20, chunk=5
    )
    check("chunking splits into 4 requests per (source, cut, arm)",
          len(chunked) == 80, str(len(chunked)))
    check("each chunk asks for 5 samples",
          all(r["body"]["n"] == 5 for r in chunked))

    check("request ids are stable across rebuilds",
          {r["req_id"] for r in reqs}
          == {r["req_id"] for r in build_requests(
              sources, model="qwen", n_cuts=6, n_continuations=20, chunk=20)})
    check("changing the model changes every id",
          not ({r["req_id"] for r in reqs}
               & {r["req_id"] for r in build_requests(
                   sources, model="other", n_cuts=6, n_continuations=20,
                   chunk=20)}))


def test_cut_plan_matches_requests():
    print("cut plan / request consistency")
    src = _source(0, "above_good")
    plan = plan_cuts(
        src.reasoning, rollout_id=src.source_id, n_cuts=6, jitter=True
    )
    reqs = build_requests([src], model="qwen", n_cuts=6)
    offsets_plan = sorted({c["char_offset"] for c in plan["cuts"] if c["t"] > 0})
    offsets_req = sorted({r["meta"]["char_offset"] for r in reqs})
    check("request offsets match the plan", offsets_plan == offsets_req,
          f"{offsets_plan} vs {offsets_req}")


def test_aggregate_counts():
    print("aggregation")
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results.jsonl"
        judged = Path(tmp) / "judged.jsonl"
        meta = {
            "source_id": "s0", "cell_id": "v1_giraffes|above_good|1000",
            "prompt_key": "v1_giraffes", "direction": "above_good",
            "threshold": 1000.0, "arm": "orig", "t": 2,
            "frac_sentences": 0.4, "n_sentences": 10,
        }
        # Three continuations: two above the threshold, one below.
        results.write_text(json.dumps({
            "req_id": "R1", "kind": "continue", "meta": meta,
            "completions": [
                {"text": "x</think>\n\n1500", "finish_reason": "stop"},
                {"text": "x</think>\n\n2000", "finish_reason": "stop"},
                {"text": "x</think>\n\n500", "finish_reason": "stop"},
                {"text": "never closed", "finish_reason": "length"},
            ],
        }) + "\n")
        lines = []
        for idx, val in enumerate(["1500", "2000", "500"]):
            lines.append(json.dumps({
                "req_id": f"J{idx}", "kind": "judge",
                "meta": {
                    "answer_key": answer_cache_key("v1_giraffes", val),
                    "parent_req_id": "R1", "completion_index": idx,
                },
                "completions": [
                    {"text": f"<final_estimate>{val}</final_estimate>"}
                ],
            }))
        judged.write_text("\n".join(lines) + "\n")

        counts = aggregate_counts(results, judged)
        check("one count cell", len(counts) == 1, str(len(counts)))
        c = counts[0]
        check("n counts only judged continuations", c["n"] == 3, str(c["n"]))
        check("k counts the favoured side", c["k"] == 2, str(c["k"]))
        check("truncated completion is flagged", c["n_truncated"] == 1,
              str(c["n_truncated"]))
        check("unjudged completion counted as unparsed", c["n_unparsed"] == 1,
              str(c["n_unparsed"]))
        check("carries frac_sentences for the model grid",
              c["frac_sentences"] == 0.4)


def test_real_tokenizer():
    print("verification against the real template (SKIP without deps)")
    for name, res in (
        ("hf apply_chat_template", verify_against_hf()),
        ("tinker Qwen3_5Renderer", verify_against_tinker()),
    ):
        if res["ok"] is None:
            print(f"  [SKIP] {name}: {res['reason']}")
        else:
            check(name, bool(res["ok"]), res["reason"])


def main() -> int:
    for fn in (
        test_prompt_shape,
        test_prefix_append,
        test_split_completion,
        test_estimate_parsing,
        test_favored_side,
        test_build_requests,
        test_cut_plan_matches_requests,
        test_aggregate_counts,
        test_real_tokenizer,
    ):
        fn()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} failure(s): {_FAILURES}")
        return 1
    print("all template/driver tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
