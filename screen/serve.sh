#!/usr/bin/env bash
# Serve Qwen3.5-35B-A3B (or the 3.6 checkpoint) via vLLM's OpenAI-compatible
# API, for the Donation Bet screen (see screen/README.md).
#
# Requires vLLM >= 0.17 -- earlier releases don't support Qwen3.5's hybrid
# GDN (gated-delta-net) + attention architecture. One 80GB H100 is the
# target; see the memory note at the bottom before you launch.
#
# Usage:
#   ./serve.sh                                  # Qwen/Qwen3.5-35B-A3B on :8000
#   ./serve.sh Qwen/Qwen3.6-35B-A3B              # the 3.6 checkpoint, :8000
#   ./serve.sh Qwen/Qwen3.6-35B-A3B 8001         # custom port
#
# This script has NOT been run against real hardware (no GPU in the dev
# environment that wrote it) -- treat flag names/values as a starting point.
# In particular, verify `--reasoning-parser` against your installed vLLM's
# `vllm serve --help` output before a real run; the flag name and the set of
# built-in parsers have moved around across vLLM releases.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-35B-A3B}"
PORT="${2:-8000}"

# --max-tokens the paper actually used for qwen3.5-35 / qwen3.6-35: 16000
# (shared/models.py _TINKER_DEFAULTS; neither model key overrides it -- only
# kimi-k2.6/kimi-k2.6-harry override to 32000, for a Tinker-specific
# truncation bug unrelated to Qwen). See README.md "Deviations" -- do not
# copy the 32000 figure into run_screen.py's --max-tokens default.
#
# --max-model-len 24576 leaves headroom for prompt (a few hundred tokens) +
# the full 16000-token completion budget + reasoning-parser overhead.
vllm serve "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --trust-remote-code

# --- Memory note (unverified against real hardware) ---
# 35B params in bf16 is ~70GB of weights alone on an 80GB H100, leaving thin
# headroom for KV cache once requests are running concurrently at
# max-model-len 24576. If vLLM OOMs at startup or under load, try in order:
#   1. --kv-cache-dtype fp8_e5m2       # ~halves KV cache memory
#   2. --max-num-seqs 16               # caps concurrent sequences
#   3. --gpu-memory-utilization 0.95   # only once (1) is also on
# run_screen.py's --max-concurrent flag (default 32) should be turned down to
# match whatever --max-num-seqs the server ends up running with, or requests
# will just queue at the server instead of erroring -- which is safe but
# slow, so it's worth tuning down for observability during the smoke test.

# --- Qwen3.6-35B-A3B variant ---
# Same architecture family and served the same way; just point at the other
# HF repo:
#   ./serve.sh Qwen/Qwen3.6-35B-A3B 8000

# --- Thinking / raw CoT ---
# run_screen.py sends `chat_template_kwargs: {"enable_thinking": true}` on
# every request and reads the parsed CoT back from
# `choice.message.reasoning_content` (the vLLM reasoning-parser convention).
# If your vLLM version doesn't populate that field for this model, drop
# --reasoning-parser here and instead have run_screen.py split the raw
# completion text on the model's own <think>...</think> tags -- see
# README.md "Deviations" for how to tell which case you're in.
