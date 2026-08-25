"""Tests for sentence cutting and prefix construction.

Run: ``python test_cuts.py`` (no pytest dependency).
"""

from __future__ import annotations

import sys

from cuts import (
    make_prefix,
    plan_cuts,
    select_cut_indices,
    split_sentences,
    strip_reasoning_markup,
)

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}")
        _FAILURES.append(label)


def test_markup_stripping():
    print("markup stripping")
    check(
        "no-op on already-parsed reasoning",
        strip_reasoning_markup("Let me think. The answer is 5.")
        == "Let me think. The answer is 5.",
    )
    check(
        "removes wrapping think tags",
        strip_reasoning_markup("<think>\nreasoning here\n</think>") == "reasoning here",
    )
    check(
        "drops answer text that leaked past the close tag",
        strip_reasoning_markup("<think>\nthinking\n</think>\n\nMy answer: 42")
        == "thinking",
    )
    check(
        "handles an unterminated block (hit the token cap)",
        strip_reasoning_markup("<think>\npartial reasoning") == "partial reasoning",
    )
    check("idempotent", strip_reasoning_markup(strip_reasoning_markup(
        "<think>\nx. y.\n</think>")) == "x. y.")
    check("empty input", strip_reasoning_markup("") == "")


def test_no_split_on_numbers_and_abbreviations():
    print("segmentation hazards")
    text = (
        "There are approx. 117,000 giraffes alive today. "
        "Each has about 200 spots (e.g. 150-250 depending on species). "
        "So 117,000 x 200 = 2.34e7 spots. "
        "Let me refine: 1.5 million is too low."
    )
    sents = split_sentences(text)
    check("four sentences, not more", len(sents) == 4, f"got {len(sents)}: "
          + " | ".join(s.text for s in sents))
    check("decimal 2.34e7 did not split", all("2.34e7" not in s.text or
          s.text.endswith("spots.") for s in sents))
    check(
        "thousands separator did not split",
        not any(s.text.strip() in {",000 giraffes alive today.", "000 x 200 = 2.34e7 spots."}
                for s in sents),
    )


def test_structure_boundaries():
    print("structural boundaries")
    text = (
        "First I list the factors:\n"
        "- population size\n"
        "- spots per animal\n"
        "\n"
        "Now the arithmetic."
    )
    sents = split_sentences(text)
    check("bullets become separate segments", len(sents) >= 4,
          f"got {len(sents)}")
    check(
        "offsets tile the text exactly",
        sents[0].start == 0 and sents[-1].end == len(text)
        and all(a.end == b.start for a, b in zip(sents, sents[1:])),
    )


def test_prefix_slicing():
    print("prefix construction")
    text = "One. Two. Three."
    sents = split_sentences(text)
    check("three sentences", len(sents) == 3, f"got {len(sents)}")
    check("prefix at t=0 is empty", make_prefix(text, 0) == "")
    check(
        "prefix at t=1 has no trailing space",
        make_prefix(text, sents[0].end) == "One.",
        repr(make_prefix(text, sents[0].end)),
    )
    check("prefix at t=n is the whole text", make_prefix(text, sents[-1].end) == text)

    newline_text = "Step one:\n- item a\n- item b"
    sents_nl = split_sentences(newline_text)
    prefix = make_prefix(newline_text, sents_nl[0].end)
    check(
        "newline boundaries keep their newline",
        prefix.endswith("\n"),
        repr(prefix),
    )


def test_cut_selection():
    print("cut selection")
    ts = select_cut_indices(20, 6)
    check("includes t=0 and t=n", ts[0] == 0 and ts[-1] == 20, str(ts))
    check("evenly spaced without jitter", ts == [0, 4, 8, 12, 16, 20], str(ts))
    check("sorted and unique", ts == sorted(set(ts)))

    short = select_cut_indices(3, 6)
    check("short CoT degrades gracefully", short == [0, 1, 2, 3], str(short))
    check("zero-sentence CoT gives only t=0", select_cut_indices(0, 6) == [0])

    check(
        "jitter never collides when the CoT is long enough",
        all(
            len(select_cut_indices(20, 6, seed=0, rollout_id=f"r{i}", jitter=True)) == 6
            for i in range(200)
        ),
    )
    a = select_cut_indices(20, 6, seed=7, rollout_id="r1", jitter=True)
    b = select_cut_indices(20, 6, seed=7, rollout_id="r1", jitter=True)
    c = select_cut_indices(20, 6, seed=7, rollout_id="r2", jitter=True)
    check("jitter is deterministic in (seed, rollout_id)", a == b, f"{a} vs {b}")
    check("jitter differs across rollouts", a != c, f"{a} vs {c}")
    check("jitter keeps the endpoints", a[0] == 0 and a[-1] == 20, str(a))
    d = select_cut_indices(20, 6, seed=8, rollout_id="r1", jitter=True)
    check("jitter responds to the seed", a != d, f"{a} vs {d}")


def test_plan_cuts():
    print("cut plans")
    reasoning = (
        "<think>\n"
        + " ".join(f"Sentence number {i} of the reasoning." for i in range(12))
        + "\n</think>"
    )
    plan = plan_cuts(reasoning, rollout_id="abc", n_cuts=6)
    check("12 sentences found", plan["n_sentences"] == 12, str(plan["n_sentences"]))
    check("six cuts", len(plan["cuts"]) == 6, str(len(plan["cuts"])))
    check("first cut is the empty prefix", plan["cuts"][0]["prefix"] == "")
    check(
        "last cut covers the whole CoT",
        plan["cuts"][-1]["prefix"] == strip_reasoning_markup(reasoning),
    )
    check(
        "prefixes are nested",
        all(
            plan["cuts"][i]["prefix"].startswith(plan["cuts"][i - 1]["prefix"])
            for i in range(1, len(plan["cuts"]))
        ),
    )
    check(
        "prefix lengths increase strictly",
        all(
            plan["cuts"][i]["prefix_chars"] > plan["cuts"][i - 1]["prefix_chars"]
            for i in range(1, len(plan["cuts"]))
        ),
    )
    check(
        "frac_sentences spans [0, 1]",
        plan["cuts"][0]["frac_sentences"] == 0.0
        and plan["cuts"][-1]["frac_sentences"] == 1.0,
    )
    check(
        "deterministic",
        plan == plan_cuts(reasoning, rollout_id="abc", n_cuts=6),
    )


def test_degenerate_inputs():
    print("degenerate inputs")
    for label, text in [
        ("empty", ""),
        ("whitespace only", "   \n\n  "),
        ("single word", "hmm"),
        ("no punctuation at all", "a very long stream of thought with no full stops"),
        ("only punctuation", "... !!! ???"),
    ]:
        plan = plan_cuts(text, rollout_id="x", n_cuts=6)
        ok = plan["cuts"][0]["prefix"] == "" and all(
            isinstance(c["char_offset"], int) for c in plan["cuts"]
        )
        check(f"{label} does not crash and starts empty", ok, str(plan))


def main() -> int:
    for fn in (
        test_markup_stripping,
        test_no_split_on_numbers_and_abbreviations,
        test_structure_boundaries,
        test_prefix_slicing,
        test_cut_selection,
        test_plan_cuts,
        test_degenerate_inputs,
    ):
        fn()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} failure(s): {_FAILURES}")
        return 1
    print("all cut tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
