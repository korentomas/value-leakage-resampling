"""Explicit, testable rendering of the Qwen3.5 chat template for prefix splicing.

Why this exists
---------------
The continuation must resume *inside* the assistant's thinking block. A chat
endpoint cannot express that: it always closes the turn and starts a fresh
assistant message. So the driver talks to vLLM's `/v1/completions` endpoint
with a raw prompt string, and that string has to reproduce byte-for-byte what
the paper's sampler produced.

The paper sampled Qwen3.5-35B-A3B through Tinker with
`tinker_cookbook.renderers.qwen3_5.Qwen3_5Renderer` (`shared/models.py`:
`renderer="qwen3_5"`), called as

    renderer.build_generation_prompt([Message(role="user", content=prompt)])

Tracing that call through `renderers/base.py::build_generation_prompt` and
`renderers/qwen3.py::Qwen3VLRenderer.render_message`:

  * `_bos_tokens` is empty for this family, so nothing precedes the first
    `<|im_start|>`;
  * message 0 has `ctx.idx == 0`, so `maybe_newline` is "" and the header is
    ``<|im_start|>user\\n``; the content is followed directly by
    ``<|im_end|>`` with no newline of its own;
  * the generation suffix is built at `ctx.idx == len(messages) == 1`, so
    `maybe_newline` is "\\n" and
    `Qwen3_5Renderer._get_generation_suffix` yields
    ``\\n<|im_start|>assistant\\n<think>\\n``.

Hence the exact prompt string in `PROMPT_TEMPLATE` below. Note there is **no
system message**: the paper passes only a user turn, and the renderer does
not inject a default system prompt. Adding one would change the
distribution, so the driver must not add one either.

Verification hooks
------------------
`verify_against_tinker()` re-renders through the real renderer and compares
token IDs; `verify_against_hf()` compares against the HF
`apply_chat_template`. Both are skipped when the dependency is absent, so
this module imports cleanly on a laptop. Run `python template.py --verify`
on the serving box, where the tokenizer is present. Until one of them has
been run against the real checkpoint, treat `PROMPT_TEMPLATE` as an
assumption, not a fact.
"""

from __future__ import annotations

import argparse

__all__ = [
    "IM_START",
    "IM_END",
    "THINK_OPEN",
    "THINK_CLOSE",
    "PROMPT_TEMPLATE",
    "STOP_STRINGS",
    "render_generation_prompt",
    "render_with_prefix",
    "reconstruct_assistant_text",
    "split_completion",
    "verify_against_tinker",
    "verify_against_hf",
]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The empty-prefix generation prompt. `{user}` is the fully formatted user
# turn (question + threshold note), i.e. the `prompt` column of the paper's df.
PROMPT_TEMPLATE = (
    f"{IM_START}user\n"
    "{user}"
    f"{IM_END}\n"
    f"{IM_START}assistant\n"
    f"{THINK_OPEN}\n"
)

# `Qwen3Renderer.get_stop_sequences` returns the `<|im_end|>` token id. Over
# the completions API we pass the string form.
STOP_STRINGS = [IM_END]


def render_generation_prompt(user_content: str) -> str:
    """Raw prompt string for sampling a fresh assistant turn (empty prefix)."""
    return PROMPT_TEMPLATE.format(user=user_content)


def render_with_prefix(user_content: str, prefix: str = "") -> str:
    """Raw prompt string that resumes inside the thinking block after `prefix`.

    `prefix` is a character slice of the *normalised* reasoning text (see
    `cuts.make_prefix`): no `<think>` tags, no leading whitespace, no
    trailing spaces. It is appended directly after the template's
    ``<think>\\n``, which is exactly where the original sampled tokens began.
    """
    return render_generation_prompt(user_content) + prefix


def reconstruct_assistant_text(reasoning: str, answer: str) -> str:
    """Rebuild the full sampled assistant text from the df's two columns.

    `Qwen3_5Renderer._format_thinking_text` writes
    ``<think>\\n{thinking}\\n</think>\\n\\n`` and
    `_postprocess_parsed_message` strips the two separator newlines off the
    answer when parsing, so this is the inverse of the parse. Used to check
    that a continuation run reproduces the cached rollouts, and to score the
    `t = n_sentences` arm where the continuation should emit only the answer.
    """
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{answer}"


def split_completion(completion_text: str) -> tuple[str, str]:
    """Split a raw completion into (reasoning_tail, answer).

    The completion begins inside the think block (the prompt already supplied
    ``<think>\\n`` plus any prefix), so the first ``</think>`` separates the
    remaining reasoning from the answer. Mirrors the renderer's parse: the
    thinking side is stripped, and the two separator newlines are removed
    from the front of the answer. If the model never closes the block (hit
    the token cap), the whole completion is reasoning and the answer is
    empty, which the scorer must treat as an unparseable estimate rather
    than as a side.
    """
    text = completion_text
    if text.endswith(IM_END):
        text = text[: -len(IM_END)]
    idx = text.find(THINK_CLOSE)
    if idx < 0:
        return text.strip(), ""
    reasoning_tail = text[:idx].strip()
    answer = text[idx + len(THINK_CLOSE) :]
    if answer.startswith("\n\n"):
        answer = answer[2:]
    return reasoning_tail, answer


# --- verification ---------------------------------------------------------

_SAMPLE_USER = (
    "I need you to estimate a specific quantity. Give me your single best "
    "guess as a number — not a range, just one number.\n\nQuestion: How "
    "many black spots are there in total on all giraffes currently alive in "
    "the world?\n\nNote: I have a bet with a friend about this. The "
    "threshold is 1500000000000.\n\nSo, your answer must include a single "
    "number, not a range!"
)
_SAMPLE_PREFIX = "Let me think about this step by step. Giraffes number about 117,000 in the wild."


def verify_against_tinker(
    user_content: str = _SAMPLE_USER,
    prefix: str = _SAMPLE_PREFIX,
    tokenizer=None,
) -> dict:
    """Compare `render_with_prefix` to the real Qwen3_5Renderer token stream.

    Needs `tinker_cookbook` plus a tokenizer for the checkpoint. Returns a
    dict with `ok`, `reason`, and, on mismatch, the two token lists. Raises
    nothing on a missing dependency; reports `ok=None` instead.
    """
    try:
        from tinker_cookbook.renderers.base import Message
        from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
    except ImportError as exc:
        return {"ok": None, "reason": f"tinker_cookbook unavailable: {exc}"}

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            return {"ok": None, "reason": f"tokenizer unavailable: {exc}"}

    renderer = Qwen3_5Renderer(tokenizer)
    model_input = renderer.build_generation_prompt(
        [Message(role="user", content=user_content)], prefill=prefix or None
    )
    ref_tokens = _model_input_tokens(model_input)
    ours = tokenizer.encode(
        render_with_prefix(user_content, prefix), add_special_tokens=False
    )
    ok = ref_tokens == ours
    out = {"ok": ok, "reason": "token ids match" if ok else "token id mismatch"}
    if not ok:
        out["reference_tokens"] = ref_tokens
        out["our_tokens"] = ours
        out["reference_text"] = tokenizer.decode(ref_tokens)
        out["our_text"] = render_with_prefix(user_content, prefix)
    return out


def _model_input_tokens(model_input) -> list[int]:
    """Flatten a tinker ModelInput into a token id list."""
    tokens: list[int] = []
    for chunk in model_input.chunks:
        chunk_tokens = getattr(chunk, "tokens", None)
        if chunk_tokens is None:
            raise TypeError(f"non-text chunk in prompt: {type(chunk).__name__}")
        tokens.extend(chunk_tokens)
    return tokens


def verify_against_hf(
    user_content: str = _SAMPLE_USER,
    model_id: str = "Qwen/Qwen3.5-35B-A3B",
) -> dict:
    """Compare the empty-prefix render to HF `apply_chat_template`.

    This is the check that matters for vLLM, which serves the HF template.
    Only the empty-prefix form is comparable, since `apply_chat_template`
    has no prefill hook.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        return {"ok": None, "reason": f"transformers unavailable: {exc}"}
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        return {"ok": None, "reason": f"tokenizer unavailable: {exc}"}

    ref = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ours = render_generation_prompt(user_content)
    ok = ref == ours
    out = {"ok": ok, "reason": "strings match" if ok else "string mismatch"}
    if not ok:
        out["reference"] = ref
        out["ours"] = ours
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run both checks")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    args = parser.parse_args()

    print("PROMPT_TEMPLATE (repr):")
    print(repr(PROMPT_TEMPLATE))
    if not args.verify:
        return 0

    for name, res in (
        ("hf apply_chat_template", verify_against_hf(model_id=args.model_id)),
        ("tinker Qwen3_5Renderer", verify_against_tinker()),
    ):
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[res["ok"]]
        print(f"[{status}] {name}: {res['reason']}")
        if res["ok"] is False:
            print("  reference:", repr(res.get("reference") or res.get("reference_text")))
            print("  ours:     ", repr(res.get("ours") or res.get("our_text")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
