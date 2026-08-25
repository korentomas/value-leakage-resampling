# swap/: cut, swap, continue

The experiment. For a cached Donation Bet rollout *i* (question *q*, direction *c*, threshold *T*), cut its CoT at a fraction *t* of its sentences and continue the prefix *N* = 25 times under three prompts:

| arm | prompt | what it measures |
|---|---|---|
| orig | *q* under condition *c* | the prefix under its own prompt |
| swap | *q* under the opposite condition, same *T* | can a contradicting prompt still move it? |
| neutral | *q* with no bet | has the text already leaned? |

```
dep_i(t) = P(good side | orig) - P(good side | swap)
s_i(t)   = P(good side | neutral) - 0.5
```

At *t* = 0 the prefix is empty and dep(0) is the paper's own bias (`donation_bet/bias_metrics.balanced_bias_score`), so the *t* = 0 row is filled from their cache and doubles as a correctness check on the pipeline (0.615 vs their 0.62).

## Files

| file | role |
|---|---|
| `cuts.py` | sentence-boundary segmentation of `reasoning`, deterministic cut selection, prefix slicing |
| `template.py` | exact Qwen3.5 chat-template rendering for prefix splicing (verified against `apply_chat_template`) |
| `driver.py` | sources -> requests -> resumable async runner (vLLM completions / judge API) -> per-cell counts |
| `model.py` | PyMC hierarchical model over dep_i(t): one smooth curve per rollout, partial pooling by question x direction, three priors, commitment summaries |
| `neutral_arm.py` | builds no-bet continuations that resume the byte-identical prefixes of the orig/swap arms |
| `three_arm.py` | s(t) and the three-gap decomposition from the (orig, neutral) fit |
| `splice_check.py` | full-trace continuation lands on the cached answer's side (20/20), validates the splice path |
| `baseline_null.py`, `null_check.py` | 4D: the null for s, prefixes cut from no-bet rollouts, continued under the no-bet prompt |
| `step4_labels.py`, `step4_awareness.py` | s(0.2) by admit/mention/deny label; the eval-awareness split |
| `prefix_direction.py` | positions of statements of intent and honesty claims in each prefix (regex, hand-checked) |
| `selfreport.py` | 4E: show Qwen its own conversation and ask whether the bet influenced it |
| `first_number.py` | 4F: the paper's trajectory-extraction prompt on each *t* = 0.2 prefix |
| `qwen36_gen.py`, `qwen36_readout.py` | 4G: the same design on Qwen3.6-35B-A3B (100 traces, cuts 0.2 and 1.0) |
| `reviewer_checks.py`, `critique_addenda.py` | robustness tables: threshold sensitivity, unparsed-continuation bounds, per-question spread |
| `synth_test.py` | synthetic recovery test for `model.py` (known commitment points and dep sizes) |
| `test_*.py` | unit and end-to-end tests; plain `python test_x.py`, no pytest |

Run order and the exact commands are in the root `README.md` under *Reproducing*. Every script reads and writes `../artifacts/`; the derived files it needs are tracked, the raw request/result dumps are not.

## Serving

vLLM on the **completions** endpoint, not chat: the continuation has to resume inside the assistant's thinking block. `template.py` renders

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{prefix}
```

which matches `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B").apply_chat_template(..., add_generation_prompt=True)` exactly (`python template.py --verify`). FP8 on one H100; sampling at temperature 1, 16k-token budget, matching the paper's Qwen rollouts.
