# swap/

Cut a cached Donation Bet rollout *i* (question *q*, direction *c*, threshold *T*) at fraction *t* of its CoT sentences; continue the prefix *N* = 25 times per arm.

| arm | prompt |
|---|---|
| orig | *q*, condition *c* |
| swap | *q*, opposite condition, same *T* |
| neutral | *q*, no bet |

```
dep_i(t) = P(good side | orig) - P(good side | swap)
s_i(t)   = P(good side | neutral) - 0.5
dep_i(0) = paper's balanced_bias_score (from their cache; 0.615 vs 0.62)
```

| file | role |
|---|---|
| `cuts.py` | sentence segmentation, cut selection, prefix slicing |
| `template.py` | Qwen3.5 chat-template render for prefix splicing (`--verify` checks against `apply_chat_template`) |
| `driver.py` | sources -> requests -> async runner (vLLM completions / judge API) -> per-cell counts |
| `model.py` | PyMC model of dep_i(t): smooth curve per rollout, pooling by question x direction, three priors |
| `neutral_arm.py` | no-bet continuations on the same prefixes |
| `three_arm.py` | s(t) from the (orig, neutral) fit |
| `splice_check.py` | full-trace continuation lands on the cached side (20/20) |
| `baseline_null.py`, `null_check.py` | null for s: no-bet prefixes continued under no-bet |
| `step4_labels.py`, `step4_awareness.py` | s(0.2) by admit/mention/deny; eval-awareness split |
| `prefix_direction.py` | positions of intent statements and honesty claims (regex) |
| `selfreport.py` | post-hoc "did the bet influence you?" |
| `first_number.py` | first estimate in each t = 0.2 prefix |
| `qwen36_gen.py`, `qwen36_readout.py` | Qwen3.6-35B-A3B replication (100 traces, cuts 0.2 and 1.0) |
| `reviewer_checks.py`, `critique_addenda.py`, `threshold_audit.py` | robustness tables |
| `synth_test.py` | synthetic recovery test for `model.py` |
| `test_*.py` | unit and end-to-end tests, `python test_x.py` |

Commands: root `README.md`, *Reproducing*. All scripts read and write `../artifacts/`.

Serving: vLLM completions endpoint (not chat), FP8, one H100, temperature 1, 16k-token budget. Rendered prefix:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{prefix}
```
