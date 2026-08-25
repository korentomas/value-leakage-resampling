# screen/: reproduce the paper's Donation Bet bias locally

Gate before the main experiment: serve Qwen3.5-35B-A3B (and Qwen3.6-35B-A3B) with vLLM and check the served model reproduces the paper's published bias (0.62 / 0.27, tolerance +/-0.15) on their own task. Every prompt template, question list, sample size, threshold routine, judge parser and bias metric is **imported** from `../external/value_leakage` (`shared`, `donation_bet`), never copied, so this harness can't drift from the paper's code. New code is only the transport: vLLM instead of Tinker for sampling, a configurable judge backend.

| file | role |
|---|---|
| `inventory.py`, `repro.py` | read-only: what the paper's cache contains, and that `get_main_dfs` reproduces 0.62 / 0.27 from it |
| `serve.sh` | vLLM launch line (FP8, `--max-model-len 24576`, reasoning parser on) |
| `run_screen.py` | sample 100 baseline + 100 per-direction rollouts per question at the baseline-median threshold |
| `judge.py` | extract the numeric estimate from each answer with the paper's judge prompt (Anthropic by default) |
| `calibrate_judge.py` | agreement of three cheaper judges with the reference judge on a 300-answer set (94-96%) |
| `bias.py` | score against the published gate; exits 1 on failure |
| `denial_labels.py`, `awareness_labels.py` | the paper's admit/mention/deny and eval-awareness labels, via their judges |
| `master_table.py` | one parquet per model joining rollouts, estimates and labels (`artifacts/qwen3.x-35_master.parquet`) |

```bash
./serve.sh                                                    # on the GPU host
python run_screen.py --model-key qwen3.5-35 --served-model Qwen/Qwen3.5-35B-A3B \
    --base-url http://<host>:8000/v1 --out ../artifacts/screen_qwen3.5-35_fp8.jsonl
python judge.py --in ../artifacts/screen_qwen3.5-35_fp8.jsonl --out ../artifacts/screen_qwen3.5-35_fp8.judged.jsonl
python bias.py  --in ../artifacts/screen_qwen3.5-35_fp8.judged.jsonl --model-key qwen3.5-35
python master_table.py
```

Result: 0.615 for Qwen3.5 in FP8 (paper: 0.62); Qwen3.6 also inside its gate.

Deviations from the paper's pipeline, all chosen to match what their Qwen rollouts were actually sampled with: `max_tokens` 16,000 (their `_TINKER_DEFAULTS`, not the 32,000 used for Kimi); temperature 1.0 with top_p/top_k unset; thinking enabled through vLLM's chat template and read back from `reasoning_content`.
