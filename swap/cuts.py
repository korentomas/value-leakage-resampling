"""Sentence-boundary cutting of CoT reasoning text for prefix-swap resampling.

The unit of a cut is a sentence boundary in the `reasoning` column of the
paper's cached dataframe (`shared/get_main_dfs.py`). A cut at index ``t``
means "keep the first ``t`` sentences"; ``t=0`` is the empty prefix and
``t=n_sentences`` is the whole CoT.

What `reasoning` actually contains
---------------------------------
For the Tinker/Qwen path (`shared/runner.py`, backend == "tinker") the text
is assembled from the parsed message's ``thinking`` parts, and
``Qwen3_5Renderer._postprocess_parsed_message`` applies ``.strip()`` to each
thinking part. So the cached `reasoning` is:

  * the *inside* of the ``<think>`` block,
  * with the ``<think>`` / ``</think>`` tags already removed,
  * whitespace-stripped at both ends.

`strip_reasoning_markup` therefore normally does nothing, but it is applied
anyway so the same code survives rollouts pulled from other backends (Claude
thinking blocks, an OpenAI reasoning summary, or a raw vLLM completion where
the tags are still present).

Assumptions, stated so they can be checked against real data
------------------------------------------------------------
A1. `reasoning` holds thinking-block content only; the post-``</think>``
    answer lives in `answer`. Splicing therefore resumes *inside* the
    thinking block (see `template.py`).
A2. Sentence segmentation is regex-based (no nltk/spacy dependency). It is
    tuned for English prose with numeric estimates: it does not split on
    decimals, thousands separators, common abbreviations, or list markers,
    and it treats blank lines and markdown headers/bullets as hard breaks.
    It will occasionally mis-split on unusual constructions; because both
    arms of the swap use the *same* cut offsets, a mis-split costs
    resolution on the t-axis, not validity of dep_i(t).
A3. A prefix is a raw character slice of `reasoning`. Trailing spaces/tabs
    are removed (a prompt ending in a space tokenizes badly under BPE), but
    trailing newlines are kept, because they carry list/paragraph structure
    that the continuation needs in order to resume in the right place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "Sentence",
    "strip_reasoning_markup",
    "split_sentences",
    "select_cut_indices",
    "cut_offsets",
    "make_prefix",
    "plan_cuts",
]


# --- markup normalisation -------------------------------------------------

_THINK_OPEN_RE = re.compile(r"<think>\s*", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"\s*</think>", re.IGNORECASE)
# Some servers emit the tags with the vLLM reasoning parser stripped; some
# emit an unterminated block. Both are handled by removing whatever is there.


def strip_reasoning_markup(text: str) -> str:
    """Remove ``<think>``/``</think>`` tags and outer whitespace.

    Idempotent, and a no-op on text that already came through the tinker
    renderer's post-processing.
    """
    if not text:
        return ""
    # Only strip the tags themselves; keep everything between them. If a
    # closing tag is present, drop anything after it (that is answer text
    # that leaked into the reasoning field).
    close = _THINK_CLOSE_RE.search(text)
    if close:
        text = text[: close.start()]
    text = _THINK_OPEN_RE.sub("", text, count=1)
    return text.strip()


# --- sentence segmentation ------------------------------------------------

# Abbreviations that end in a period but do not end a sentence. Matched
# case-insensitively against the token immediately before the period.
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "approx", "est", "no", "fig", "eq",
    "al", "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "inc", "ltd",
    "co", "corp", "dept", "univ", "min", "max", "avg", "sec", "hr", "hrs",
    "yr", "yrs", "km", "cm", "mm", "kg", "lb", "ft", "in", "ca", "u.s",
    "u.k", "p.a", "b.c", "a.d", "a.m", "p.m", "mil", "bn", "tn",
}

_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*")
_HARD_BREAK_RE = re.compile(r"\n\s*\n")
# A line that begins a markdown block: heading, bullet, or numbered item.
_BLOCK_START_RE = re.compile(r"\n(?=\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s))")

_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


@dataclass(frozen=True)
class Sentence:
    """One segment of the reasoning text.

    `start` / `end` are character offsets into the *normalised* reasoning
    string (the output of `strip_reasoning_markup`), with `end` pointing one
    past the final character of the segment including its trailing
    whitespace. `text` is the segment with surrounding whitespace stripped.
    """

    index: int
    start: int
    end: int
    text: str


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    """True if the period at `dot_pos` closes a known abbreviation."""
    m = _WORD_BEFORE_RE.search(text[:dot_pos])
    if not m:
        return False
    return m.group(1).rstrip(".").lower() in _ABBREVIATIONS


def _is_decimal_or_grouped_number(text: str, dot_pos: int) -> bool:
    """True if the period at `dot_pos` sits inside a number (3.5, 1.2e9)."""
    if dot_pos + 1 >= len(text):
        return False
    return text[dot_pos - 1 : dot_pos].isdigit() and text[dot_pos + 1].isdigit()


def _is_initial(text: str, dot_pos: int) -> bool:
    """True for a single-letter initial such as the `A.` in `A. Smith`."""
    if dot_pos < 1:
        return False
    if not text[dot_pos - 1].isupper():
        return False
    return dot_pos == 1 or not text[dot_pos - 2].isalpha()


def _boundary_positions(text: str) -> list[int]:
    """Character offsets just past each sentence-final punctuation run."""
    out = []
    for m in _SENT_END_RE.finditer(text):
        dot_pos = m.start()
        if text[dot_pos] == "." and (
            _is_decimal_or_grouped_number(text, dot_pos)
            or _is_abbreviation(text, dot_pos)
            or _is_initial(text, dot_pos)
        ):
            continue
        end = m.end()
        # Require the boundary to be followed by whitespace or end-of-text;
        # "3.5x" and "U.S.A" style runs otherwise generate spurious cuts.
        if end < len(text) and not text[end].isspace():
            continue
        out.append(end)
    return out


def split_sentences(reasoning: str, *, normalise: bool = True) -> list[Sentence]:
    """Segment `reasoning` into sentences at deterministic offsets.

    Boundaries are placed at sentence-final punctuation (skipping decimals,
    abbreviations, and initials), at blank lines, and before markdown block
    starts. Consecutive whitespace after a boundary belongs to the preceding
    segment, so `Sentence.end` values tile the string exactly.
    """
    text = strip_reasoning_markup(reasoning) if normalise else reasoning
    if not text:
        return []

    cuts = set(_boundary_positions(text))
    cuts.update(m.start() for m in _HARD_BREAK_RE.finditer(text))
    cuts.update(m.start() + 1 for m in _BLOCK_START_RE.finditer(text))
    cuts.discard(0)

    sentences: list[Sentence] = []
    prev = 0
    for pos in sorted(cuts) + [len(text)]:
        # Absorb the whitespace that follows the boundary into this segment
        # so the next segment starts on a non-space character.
        end = pos
        while end < len(text) and text[end].isspace():
            end += 1
        seg = text[prev:end].strip()
        if not seg:
            continue
        sentences.append(Sentence(len(sentences), prev, end, seg))
        prev = end
    if prev < len(text) and text[prev:].strip():
        sentences.append(
            Sentence(len(sentences), prev, len(text), text[prev:].strip())
        )
    return sentences


# --- cut selection --------------------------------------------------------


def _seed_int(seed, *parts: str) -> int:
    """Stable 64-bit integer from a seed plus identifying strings."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode())
    return int.from_bytes(h.digest(), "big")


def select_cut_indices(
    n_sentences: int,
    n_cuts: int = 6,
    *,
    seed: int = 0,
    rollout_id: str = "",
    jitter: bool = False,
) -> list[int]:
    """Choose ~`n_cuts` sentence indices, evenly spaced, including 0 and n.

    Returns a sorted, de-duplicated list of cut indices ``t`` in
    ``[0, n_sentences]``. ``t=0`` (empty prefix, pure prompt effect) and
    ``t=n_sentences`` (whole CoT) are always included when ``n_cuts >= 2``.

    Selection is deterministic. With ``jitter=True`` the interior cuts are
    displaced by at most half a spacing using a hash of
    ``(seed, rollout_id)``, which decorrelates cut positions across rollouts
    so that sentence-length artefacts do not line up across the corpus. The
    same ``(seed, rollout_id)`` always yields the same cuts, and both arms of
    a swap share them by construction.
    """
    if n_sentences < 0:
        raise ValueError("n_sentences must be non-negative")
    if n_cuts < 1:
        raise ValueError("n_cuts must be at least 1")
    if n_sentences == 0:
        return [0]
    if n_cuts == 1:
        return [0]

    fractions = [i / (n_cuts - 1) for i in range(n_cuts)]
    raw = [f * n_sentences for f in fractions]

    if jitter and n_cuts > 2:
        rng_state = _seed_int(seed, rollout_id)
        span = n_sentences / (n_cuts - 1)
        for i in range(1, n_cuts - 1):
            # Deterministic per-(rollout, cut) offset in +/- 0.35 spacings.
            # Kept below half a spacing so jittered cuts stay in their own
            # fraction bin and rarely collide with a neighbour.
            bits = (rng_state >> (8 * i)) & 0xFF
            raw[i] += ((bits / 256.0) - 0.5) * 0.7 * span

    ts = [min(n_sentences, max(0, int(round(x)))) for x in raw]

    # Separate collisions when the CoT is long enough to hold `n_cuts`
    # distinct boundaries; otherwise a short CoT simply yields fewer cuts.
    if n_sentences >= n_cuts - 1:
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                ts[i] = ts[i - 1] + 1
        ts[-1] = n_sentences
        for i in range(len(ts) - 2, 0, -1):
            if ts[i] >= ts[i + 1]:
                ts[i] = ts[i + 1] - 1
    return sorted({t for t in ts if 0 <= t <= n_sentences})


def cut_offsets(sentences: list[Sentence], ts: list[int]) -> list[int]:
    """Character offsets corresponding to cut indices `ts`."""
    offsets = []
    for t in ts:
        if t <= 0:
            offsets.append(0)
        elif t >= len(sentences):
            offsets.append(sentences[-1].end if sentences else 0)
        else:
            offsets.append(sentences[t - 1].end)
    return offsets


def make_prefix(reasoning: str, offset: int, *, normalise: bool = True) -> str:
    """Character slice of the normalised reasoning, safe to append to a prompt.

    Trailing spaces and tabs are removed so the prompt does not end mid-token
    (BPE merges the leading space into the *next* token); trailing newlines
    are kept because they carry block structure the continuation must resume
    inside.
    """
    text = strip_reasoning_markup(reasoning) if normalise else reasoning
    return text[:offset].rstrip(" \t")


def plan_cuts(
    reasoning: str,
    *,
    rollout_id: str,
    n_cuts: int = 6,
    seed: int = 0,
    jitter: bool = False,
) -> dict:
    """Full cut plan for one source rollout.

    Returns ``{"n_sentences", "cuts": [{"t", "char_offset", "prefix",
    "prefix_chars", "frac_sentences"}, ...]}``. Deterministic in
    ``(reasoning, rollout_id, n_cuts, seed, jitter)``.
    """
    text = strip_reasoning_markup(reasoning)
    sentences = split_sentences(text, normalise=False)
    n = len(sentences)
    ts = select_cut_indices(
        n, n_cuts, seed=seed, rollout_id=rollout_id, jitter=jitter
    )
    offsets = cut_offsets(sentences, ts)
    cuts = []
    for t, off in zip(ts, offsets):
        prefix = make_prefix(text, off, normalise=False)
        cuts.append(
            {
                "t": t,
                "char_offset": off,
                "prefix": prefix,
                "prefix_chars": len(prefix),
                "frac_sentences": (t / n) if n else 0.0,
            }
        )
    return {"n_sentences": n, "cuts": cuts}
