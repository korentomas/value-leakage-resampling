# screen/

Gate: served Qwen3.5-35B-A3B / Qwen3.6-35B-A3B must reproduce the paper's Donation Bet bias (0.62 / 0.27, tolerance ±0.15). Prompts, questions, thresholds, judge parser and bias metric are imported from `../external/value_leakage`; only the transport is new (vLLM instead of Tinker, configurable judge).

| file | role |
|---|---|
| `inventory.py`, `repro.py` | read the paper's cache; `get_main_dfs` gives 0.62 / 0.27 |
| `serve.sh` | vLLM launch (FP8, `--max-model-len 24576`, reasoning parser) |
| `run_screen.py` | 100 baseline + 100 per-direction rollouts per question |
| `judge.py` | numeric estimate per answer, paper's judge prompt |
| `calibrate_judge.py` | three cheaper judges vs reference on 300 answers (94–96% agreement) |
| `bias.py` | score against the gate; exit 1 on failure |
| `denial_labels.py`, `awareness_labels.py` | paper's admit/mention/deny and eval-awareness labels, their judges |
| `master_table.py` | `artifacts/<model>_master.parquet`: rollouts + estimates + labels |

Result: 0.615 (Qwen3.5, FP8). Sampling matches the paper's Qwen rollouts: `max_tokens` 16000, temperature 1, top_p/top_k unset, thinking via chat template.
