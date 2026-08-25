# Infra Recon: Serving Qwen3.5-35B-A3B / Qwen3.6-35B-A3B on RunPod (vLLM)

Read-only recon. No RunPod resources were created, started, stopped, or modified.
Date: 2026-08-17.

---

## 1. Model identity

| | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| Repo id | `Qwen/Qwen3.5-35B-A3B` (also `-Base`) | `Qwen/Qwen3.6-35B-A3B` |
| Total params | 35,951,822,704 (~36.0B) | 35,951,822,704 (~36.0B) |
| Active params/token | ~3B | ~3B |
| License | Apache 2.0 | Apache 2.0 |
| Context (native / YaRN) | 262,144 / 1,010,000 | 262,144 / 1,010,000 |
| FP8 variant | not directly confirmed on HF (check before use) | `Qwen/Qwen3.6-35B-A3B-FP8`, confirmed, Apache 2.0 |
| NVFP4 variant |, | `nvidia/Qwen3.6-35B-A3B-NVFP4` (Blackwell only) |
| GGUF/quantized community | 280 quant repos | 754 quant repos (llama.cpp/LM Studio/Jan/Ollama) |

**Checkpoint size on disk** (from HF `safetensors` metadata, `Qwen/Qwen3.6-35B-A3B`, same param count applies to 3.5):
- **BF16 full checkpoint:** 35.95B params x 2 bytes ~ **71.9 GB**
- **FP8 checkpoint** (`Qwen3.6-35B-A3B-FP8`): 1.50B params kept in BF16 (embeddings/norms) + 34.45B params in FP8 ~ **37.5 GB**

**Architecture**, confirmed hybrid design, 40 layers total, organized as 10 cycles of `3x (Gated DeltaNet -> MoE) -> 1x (Gated Attention -> MoE)`:
- 30 GatedDeltaNet (linear-attention) layers: fixed-size recurrent state, not a growing KV cache, this is the memory-saving property that matters for our long-shared-prefix workload.
- 10 Gated Attention layers (standard, quadratic): 16 Q heads / 2 KV heads, 256 head dim, 64-dim RoPE, this matches your "10/40 standard attention layers" description exactly.
- MoE FFN on every block: 256 experts, top-8 routed + 1 shared, 512-dim expert intermediate.
- Practical consequence: KV-cache memory footprint per sequence is much smaller than a 40-layer dense transformer would need, because only 10 of 40 layers hold a real KV cache, this buys headroom for either longer context or higher batch concurrency on a single 80GB card.

Qwen3.5-35B-A3B-Base and Qwen/Qwen3.5-35B-A3B share the same param count and appear to be the architectural predecessor generation (same layer layout described on its model card); treat as functionally identical for infra sizing purposes.

Sources: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)

---

## 2. vLLM support

- **Minimum version:** vLLM >= 0.19.0 recommended by the Qwen team specifically for Qwen3.6 (`--reasoning-parser qwen3` needs this). vLLM recipes list >=0.17.0 as a floor for standard deployment, >=0.24.0 only if you want NVFP4 on DGX Spark (not our path). **Recommendation: pin to latest stable vLLM (>=0.19) rather than the floor**, it also gets you the native hidden-states extraction feature (see Sec. 5), which landed as of ~v0.18.
- **GPU memory / TP requirement** (per official vLLM recipe, recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B):
  - BF16: needs 1x H200 (141GB) or 2x H100 (tensor-parallel), a single 80GB H100 does **not** fit the 71.9GB BF16 checkpoint with any usable headroom.
  - **FP8: fits a single H100/H200** (checkpoint is 37.5GB), this is the path for a one-card deployment.
  - NVFP4: Blackwell-only, not relevant to RunPod's current H100/A100 fleet.
- **Reference serve command (single H100, FP8):**
  ```
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --max-model-len 32768 --reasoning-parser qwen3
  ```
  (drop `--max-model-len` down from the recipe's 262144 default since our workload only needs ~32k context; this also reduces KV-cache/state memory reservation and leaves more room for concurrent sequences).
- **Prefix caching with the hybrid arch:** confirmed **experimental**. The recipe states: "Prefix Caching (Mamba): currently experimental in 'align' mode", i.e., caching across the GatedDeltaNet recurrent state is not yet a solid, production-grade path. Two known separate issues to watch for: (a) a CUDA-graph capture assertion (`num_cache_lines >= batch`) that requires reducing `--max-cudagraph-capture-size` below the default 512 if it fires, and (b) the align-mode caching itself being flagged experimental rather than default-safe.
  - **Practical read for your workload:** because prefill throughput on this class of model is very high (tens of thousands of tok/s per stream even at 8-32k context, see Sec. 4), recomputing a shared prefix instead of caching it is *not* the dominant cost unless your shared prefix is very long (tens of thousands of tokens) and resent an extreme number of times. For 9 questions x 300 rollouts with prefixes in the low-thousands-of-tokens range, expect prefix recompute to be a minor fraction of total wall-clock even with caching disabled, don't block on getting prefix caching working, but do a small pilot to confirm before trusting this at your actual prefix length.
- **`/v1/completions` (raw prompt, not chat template):** not explicitly confirmed either way in the docs pulled. vLLM's OpenAI-compatible server exposes `/v1/completions` for every model by default (it's not model-specific), so this should work; the recipe docs just don't call it out because they demo `/v1/chat/completions`. Verify with a smoke test before relying on it, if you need raw prompts without the chat template/reasoning-parser wrapping, this is the one item worth confirming empirically first.

Sources: [Qwen/Qwen3.6-35B-A3B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B), [Qwen3.5 & Qwen3.6 usage guide](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)

---

## 3. RunPod: GPU availability, pricing, config recommendation

Live catalog snapshot (read-only, `list-gpu-types` / `get-gpu-type`):

| GPU | VRAM | Community $/hr | Secure $/hr | Aggregate availability | Per-DC availability (sampled) |
|---|---|---|---|---|---|
| H100 SXM (HBM3) | 80GB | $2.69 | $3.29 | HIGH | all sampled DCs show LOW individually (stock spread thin across ~10 DCs) |
| H100 NVL | 94GB | $2.59 | $3.19 | LOW |, |
| H100 PCIe | 80GB | $1.99 | $2.89 | **NONE (out of stock)** |, |
| A100 SXM | 80GB | $1.39 | $1.59 | HIGH |, |
| A100 PCIe | 80GB | $1.19 | $1.39 | LOW |, |
| H200 SXM | 141GB | $3.59 | $4.59 | MEDIUM |, |
| H200 NVL | 143GB | $0.50 (community, likely mispriced/promo, verify) | $3.79 | LOW |, |

Note: the aggregate "HIGH" for H100 SXM vs. every individual datacenter reporting "LOW" is not a contradiction, it means stock exists but is thinly spread across many regions, not concentrated in one place. **Re-check live capacity at actual provisioning time**, this is a moving number.

### Recommended config

- **GPU: H100 SXM 80GB, secure cloud, single card.** FP8 checkpoint (37.5GB) + vLLM/CUDA overhead leaves ~35-40GB free for KV cache (only 10/40 layers need real KV) + GatedDeltaNet recurrent state + activation buffers, comfortable at 32k context with meaningful batch concurrency.
  - Secure cloud ($3.29/hr) over community ($2.69/hr) unless cost pressure is severe, secure cloud avoids the eviction/preemption risk that community-cloud pods carry, which matters for a multi-hour unattended generation job. If you want to shave cost and can tolerate restart risk, community cloud is 18% cheaper.
  - A100 SXM ($1.59/hr secure) is **not** viable for FP8 alone at full speed, A100 lacks native FP8 tensor-core support at H100's throughput, so FP8 kernels either fall back to slower paths or aren't optimal; A100 is really a BF16-class card. Given the 71.9GB BF16 checkpoint doesn't fit an 80GB A100 with headroom either, A100 is not a good fit for this model without a 2-card TP setup, which erases its price advantage. Stick with H100 SXM.
- **Image:** official RunPod template "Runpod Pytorch 2.8.0" (`runpod-torch-v280`) as base, then `pip install "vllm>=0.19"` on top. (No pre-built vLLM-serving template was found under this account via `list-templates`; building on the PyTorch base is the standard path.)
- **Disk:**
  - Container/volume needs: FP8 weights (~40GB with overhead) + vLLM/deps (~10GB) + ~50GB rollout JSONL (your estimate) + buffer => **recommend a 150-200GB persistent network volume**, not just container disk. Attaching weights + outputs to a network volume means you don't re-download the 37.5GB checkpoint every time a pod restarts, and outputs survive pod teardown.
  - Pick a datacenter that has **both** H100 SXM stock and network-volume support (STANDARD or HIGH_PERFORMANCE), cross-check at provisioning time since the two lists don't perfectly overlap in what was sampled here (e.g., `US-CA-2` and `CA-MTL-4` offer HIGH_PERFORMANCE volumes; H100 SXM per-DC stock wasn't confirmed at those specific two in this recon, verify together at deploy time, RunPod stock shifts constantly).

### Cost estimate against a $300 ceiling

Two workloads, combined 50 GPU-hours as scoped:

| Phase | GPU-hours | Secure ($3.29/hr) | Community ($2.69/hr) |
|---|---|---|---|
| (a) Screening (9x300 rollouts) | ~20 | $65.80 | $53.80 |
| (b) Continuations (~150k short gens) | ~30 | $98.70 | $80.70 |
| **Total** | **50** | **$164.50** | **$134.50** |

Plus a modest allowance for network volume storage (~$0.05-0.10/GB-month on RunPod-class pricing; a 200GB volume for a few days of active work is on the order of a few dollars) and any pilot/debug time. **All-in estimate: ~$150-200**, leaving $100-150 of margin against the $300 ceiling, see Sec. 4 for why this holds even under a conservative throughput assumption.

---

## 4. Throughput estimate and wall-clock

Only hard benchmark found for this model class on H100: [Qwen3.5-35B-A3B FP8 on 1x H100 SXM, Millstone AI](https://www.millstoneai.com/inference-benchmark/qwen3-5-35b-a3b-fp8-1x-h100-sxm) (no prompt caching, no speculative decoding, full-precision KV cache, i.e. a worst-case/conservative baseline):

- Single-stream decode: 212.2 tok/s @ 1K context -> 142.9 tok/s @ 256K context
- Batch decode (10 concurrent, 1K context): **907.7 tok/s aggregate** (peak observed)
- Batch decode (10 concurrent, 32K context): 509.9 tok/s aggregate
- Prefill: 57,296 tok/s (single request, 8K context) down to 37,856 tok/s (10 concurrent, 32K context)
- No data point beyond 10 concurrent requests was available, this is the main throughput unknown.

**(a) Screening workload:** 9 questions x 300 rollouts x ~2-5k thinking tokens => ~2,700 rollouts, say ~3,500 tokens average => **~9.45M output tokens**. At the conservative 907 tok/s batch figure: 9.45M/907 ~ 2.9 hours. Even at the high end of the thinking-token estimate (5k avg, 13.5M tokens): ~ 4.1 hours. Both comfortably inside the ~20 GPU-hour screening budget, with prefill and batching overhead easily absorbed.

**(b) Continuations workload:** ~150,000 continuations x ~1k tokens => **~150M output tokens**.
- At the benchmarked 907 tok/s (only tested up to 10 concurrent requests): 150M/907 ~ **45.9 hours**, this exceeds the ~30 GPU-hour budget.
- Short continuations (1k tokens, modest prompt) are exactly the case where pushing concurrency well past 10 should pay off: GatedDeltaNet's fixed-size state keeps per-sequence memory low, so KV/state memory is not the limiter at 32 GB+ of headroom, and MoE decode at high batch is closer to bandwidth-bound on reading FP8 weights than on any single sequence's cost. Vendors commonly report 2,000-4,000+ tok/s aggregate for A3B-class MoE models under large-batch offline serving on H100, but **I have no direct citation for that range on this specific model**, treat it as an informed extrapolation, not a confirmed number.
- **Budget sensitivity is favorable even in the worst case:** using only the benchmarked 907 tok/s for the entire continuations phase (no concurrency improvement at all) costs ~45.9 hours ~ $151-184 depending on cloud tier. Added to the ~$11-14 for screening, total lands at **~$165-197 even under the pessimistic assumption**, still under $300.
- **Recommendation:** before committing the full continuation budget, run a small pilot (500-1,000 continuations) at your intended `--max-num-seqs` concurrency (try 64-128, tuned up from the benchmark's 10) to measure actual aggregate tok/s on your prompt distribution, then extrapolate hours/cost from that real number rather than the single-concurrency-10 data point.

---

## 5. Activation-caching feasibility (residual-stream access)

**Serving through vLLM's standard OpenAI-compatible endpoint does not expose residual-stream activations**, confirmed, this remains true. But there are now two paths inside the vLLM ecosystem worth knowing about, both landed relatively recently:

1. **vLLM native hidden-state extraction** (`docs.vllm.ai/.../extract_hidden_states`), merged as of ~v0.18, originally built for EAGLE-style draft-model training. Enabled via `speculative_config={"method": "extract_hidden_states", ...}` (offline) or `--speculative_config` (serving). Layer selection via `eagle_aux_hidden_state_layer_ids` (e.g. `[1,2,3,4]`, or `num_hidden_layers` for the final layer). **Known constraint: chunked prefill must be disabled** when this is active. No documentation found on FP8 or MoE/hybrid-attention compatibility either way, unconfirmed for this model.
2. **vllm-lens** ([UKGovernmentBEIS/vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens)), purpose-built interpretability tool, auto-registers as a vLLM plugin, captures residual-stream hidden states from arbitrary layers (`output_residual_stream: [15, 20]`), supports steering vectors and custom hooks. Requires `enforce_eager=True` (**CUDA graphs disabled**, which will meaningfully cut throughput). No MoE or GatedDeltaNet-hybrid model in their validated examples (they validate GLM-4.5-Air/GLM-5.2, both dense-attention models), **not confirmed compatible with Qwen3.5/3.6's architecture**, though residual-stream hooks operate on layer *output* tensors, which should in principle be architecture-agnostic (they don't reach into the GatedDeltaNet internals, just the hidden_states between blocks), this is a reasonable expectation, not a tested fact.

**Recommended split, given the uncertainty:**
- Run the two throughput-critical phases (screening, continuations) on standard vLLM serving with CUDA graphs on, do not touch activation extraction there, it would tank throughput and blow the budget.
- For the subset of rollouts you actually need activations for, run a **separate, smaller pass**: either vllm-lens with `enforce_eager=True` on the same H100 (slower per-token but fine for a subset), or fall back to plain HF `transformers` (`output_hidden_states=True`) on a single A100/H100, the model card explicitly states Transformers compatibility, so this is the lowest-risk fallback if vllm-lens turns out incompatible with the hybrid arch.
- **Before committing to either tool for real work, run a 2-3 prompt smoke test first**, this is the single least-confirmed item in this whole recon (nobody has documented GatedDeltaNet + hidden-state-extraction compatibility either way), and a smoke test costs minutes, not the alternative of discovering it doesn't work after a long paid run.

---

## Executive summary (10 lines)

1. Both models are real, current (Aug 2026) releases: `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`, Apache 2.0, ~36B total / ~3B active params, 40-layer hybrid arch (30 GatedDeltaNet linear-attention layers + 10 standard Gated Attention layers + MoE FFN), 262K native context.
2. FP8 checkpoint (`Qwen3.6-35B-A3B-FP8`) is 37.5GB on disk and fits a single 80GB H100 with room to spare; BF16 (71.9GB) does not fit one H100, needs 2xH100 TP or 1xH200 per the official vLLM recipe.
3. vLLM >=0.19.0 recommended (`--reasoning-parser qwen3`); prefix caching for the Mamba/GatedDeltaNet cache is still experimental, but prefill is cheap enough on this model class that caching isn't load-bearing for your prefix lengths, verify empirically, don't block on it.
4. **Recommended config: 1x H100 SXM 80GB, secure cloud ($3.29/hr), FP8 checkpoint, `--max-model-len 32768`, on a RunPod PyTorch 2.8.0 base image + `pip install vllm`, with a 150-200GB persistent network volume for weights + JSONL outputs.**
5. Screening (9x300 rollouts, ~3,500 avg output tokens) needs only ~3-4 GPU-hours against a 20-hour budget, comfortable margin.
6. Continuations (~150kx1k tokens = 150M tokens) is the cost driver: worst case (907 tok/s, the only benchmarked concurrency-10 number) is ~46 GPU-hours, exceeding the 30-hour estimate but likely improvable well past that with higher `--max-num-seqs`; pilot 500-1,000 continuations first to calibrate real throughput before committing the full run.
7. **Total infra cost estimate: ~$150-200, even under the pessimistic throughput assumption**, comfortably under the $300 ceiling with $100+ margin.
8. Activation/residual-stream extraction is not natively exposed by vLLM's serving endpoint; two newer options exist (vLLM's native `extract_hidden_states` speculative-decoding path, and the purpose-built `vllm-lens` plugin), but **neither has confirmed compatibility with this hybrid GatedDeltaNet+MoE architecture**, treat as unverified and smoke-test on 2-3 prompts before relying on it.
9. Cheapest workable split: keep the high-throughput generation phases on plain vLLM with CUDA graphs on; run activation extraction as a separate small pass (vllm-lens with `enforce_eager=True`, or plain HF `transformers` as the safe fallback) on just the rollout subset you need probed.
10. Live GPU stock is currently favorable (H100 SXM aggregate HIGH, though thinly spread per-datacenter), re-check `get-capacity`/`get-gpu-type` at actual provisioning time since stock is a moving target.

---

## 6. Follow-up: BF16 screening plan (decision: BF16 for screening to avoid a quantization confound vs. the paper's BF16 Tinker sampling; FP8 stays for continuations)

Read-only follow-up, 2026-08-17. Ground truth for the arithmetic below is `Qwen/Qwen3.6-35B-A3B/config.json` (`text_config`), fetched directly, not inferred from prose. Qwen3.5-35B-A3B's config.json has the identical `num_hidden_layers`, `full_attention_interval`, head counts, and head dims (checked side-by-side), so the same arithmetic applies to either model unchanged.

### 6.1 Confirmed layer/head numbers from config.json

- `num_hidden_layers`: 40, `full_attention_interval`: 4 -> layer pattern is 3xlinear_attention then 1xfull_attention, repeated 10x. **10 full-attention layers, 30 GatedDeltaNet (linear_attention) layers.** Matches Sec. 1's description exactly, now confirmed from the source config rather than a model-card summary.
- Full attention layers: `num_attention_heads` (Q) = 16, `num_key_value_heads` (KV) = 2, `head_dim` = 256.
- GatedDeltaNet layers: `linear_num_key_heads` = 16, `linear_key_head_dim` = 128, `linear_num_value_heads` = 32, `linear_value_head_dim` = 128, `linear_conv_kernel_dim` = 4, `mamba_ssm_dtype`: **float32** (the recurrent state is kept in fp32 regardless of model dtype, matters for the size math below).
- `hidden_size`: 2048, `moe_intermediate_size`: 512, 256 experts / top-8 + 1 shared, `max_position_embeddings`: 262144.

TP=2 shards cleanly on every one of these head counts (16, 2, 16, 32 all divide by 2 with no remainder), which matters for Sec. 6.4 below.

### 6.2 KV cache + GDN state arithmetic (per sequence, BF16, `--max-model-len 32768`)

**Full-attention KV cache** (grows with context length, standard GQA):
`2 (K+V) x num_key_value_heads(2) x head_dim(256) x 2 bytes(BF16) = 2,048 bytes/token/layer`
`x 10 full-attention layers = 20,480 bytes/token`
`x 32,768 tokens (worst case, sequence fills max-model-len) = 671,088,640 bytes ~ 0.671 GB/sequence`

**GatedDeltaNet recurrent state** (fixed size, independent of context length, this is the whole point of the linear-attention design):
`num_value_heads(32) x key_head_dim(128) x value_head_dim(128) x 4 bytes (fp32, per mamba_ssm_dtype) = 2,097,152 bytes/layer ~ 0.0021 GB/layer`
plus a small causal-conv state (`linear_conv_kernel_dim - 1 = 3` cached timesteps over the ~8,192-wide fused QKV-gate projection, fp32) ~ 0.0001 GB/layer, negligible next to the recurrent state.
`x 30 GatedDeltaNet layers ~ 0.066 GB/sequence`, **flat regardless of how long the sequence runs.**

So per concurrent sequence at full 32,768-token context: **~ 0.671 (attention KV) + 0.066 (GDN state) ~ 0.74 GB**, dominated by the 10 real-KV layers, not the 30 linear-attention ones, confirms the memory-saving claim from Sec. 1 quantitatively.

### 6.3 Does BF16 fit a single H100 NVL (94GB)?

| Component | @ `--max-num-seqs 8` | @ `--max-num-seqs 16` |
|---|---|---|
| Weights (BF16) | 71.90 GB | 71.90 GB |
| Per-seq KV+state x N seqs (worst case, all at 32,768 ctx) | 8 x 0.74 = 5.9 GB | 16 x 0.74 = 11.9 GB |
| vLLM/CUDA overhead (activations, PyTorch/CUDA context, block tables), estimated, not benchmarked | ~3-5 GB | ~3-5 GB |
| **Total** | **~80.8-82.8 GB** | **~86.8-88.8 GB** |
| **Headroom on 94GB card** | **~11-13 GB** | **~5-7 GB** |

**Verdict: fits, but tight at the top of the requested range.** `--max-num-seqs 8` leaves a comfortable ~11-13GB margin; `--max-num-seqs 16` is workable but leaves only ~5-7GB, which is thin once you account for vLLM's default `--gpu-memory-utilization 0.9` reserving only 84.6GB of the 94GB as budget in the first place (0.9 x 94 = 84.6GB, below the ~86.8GB needed at max-num-seqs 16). **Recommendation: run `--max-num-seqs 8` as the safe default, or bump `--gpu-memory-utilization` to 0.95 (89.3GB budget) if you need 16.** Either way, cap `--max-cudagraph-capture-size` at or below whatever `--max-num-seqs` you choose, the recipe's documented `assert num_cache_lines >= batch` failure (Sec. 2) fires exactly when CUDA-graph capture batch sizes exceed the number of allocated GDN state slots, and the default capture size (512) massively exceeds a max-num-seqs of 8-16.

### 6.4 H100 NVL stock (per-datacenter, `get-gpu-type`)

Aggregate availability: **LOW** (weaker than the H100 SXM picture in Sec. 3). Only 2 datacenters currently carry it at all: **US-GA-2** and **US-KS-2**, both individually LOW. $2.59/hr community, $3.19/hr secure, 94GB, up to 8 cards/secure pod. This is thinner stock than H100 SXM (10 datacenters, aggregate HIGH), **given the tight memory fit in Sec. 6.3 plus thin stock, 2xH100 SXM TP=2 is the more reliable path for the actual screening run; treat single-card NVL as an opportunistic option if it happens to be available, not the primary plan.**

### 6.5 TP=2 caveats for the hybrid GatedDeltaNet arch (searched vLLM issue tracker directly, checked live status via `gh api`)

- **`in_proj_ba` Marlin `MIN_THREAD_N=64` failure** ([#35924](https://github.com/vllm-project/vllm/issues/35924), closed 2026-05-21, fixed): hits GatedDeltaNet's fused B/A gating projection when GPTQ/Marlin quantization is combined with TP splitting the projection below Marlin's 64-element minimum. **Only affects GPTQ/Marlin-quantized weights, does not apply to plain BF16 serving**, so irrelevant to this plan, but worth remembering if you ever TP a quantized GDN checkpoint.
- **FLA linear-attention tensor-format mismatch / gibberish output** ([#38643](https://github.com/vllm-project/vllm/issues/38643), still nominally **open** but stale): originally reported on a nightly build ~750 commits behind current main; the root-cause warning was a false positive removed by #38255. Two independent reporters in June-July 2026 could not reproduce it on current main with BF16 checkpoints, including one using it successfully with **vllm-lens** (relevant to Sec. 5's activation-caching question, this is a small positive data point that vLLM's GDN/FLA path and vllm-lens coexist in practice, though still not confirmed on this specific 35B-A3B checkpoint). **Practical takeaway: pin to a reasonably current vLLM build (well past ~April 2026) and this class of bug should not appear.**
- **V2 model runner KV-cache-init crash** ([#38041](https://github.com/vllm-project/vllm/issues/38041), still open): only triggers when `VLLM_USE_V2_MODEL_RUNNER=1` is explicitly set, this is opt-in, not the default. **Action: do not set this env var.**
- No open issue found specifically about plain BF16 tensor-parallel=2 correctness for this architecture, the clean head-count divisibility in Sec. 6.1 (16/2, 2/2, 16/2, 32/2, all integer) also means TP=2 avoids the whole class of "undersized shard" bugs that surfaces at higher TP degrees.

### 6.6 Serve commands

**(A) Single H100 NVL, BF16 (opportunistic, only if stock is up):**
```
vllm serve Qwen/Qwen3.6-35B-A3B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --max-cudagraph-capture-size 8 \
  --reasoning-parser qwen3
```
`--dtype bfloat16` makes the no-quantization invariant explicit in the command rather than relying on the checkpoint default. `--max-num-seqs 8` and `--gpu-memory-utilization 0.95` keep total usage (~80.8GB weights+cache+overhead against an 89.3GB budget) safely inside the tight 94GB card per Sec. 6.3. `--max-cudagraph-capture-size 8` matches `--max-num-seqs` to sidestep the documented `num_cache_lines >= batch` assertion.

**(B) 2x H100 SXM, tensor-parallel BF16 (recommended primary path, better stock, clean TP=2 shard, comfortable headroom):**
```
vllm serve Qwen/Qwen3.6-35B-A3B \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3
```
TP=2 divides every relevant head count evenly (Sec. 6.1), avoiding the shard-size bug class seen at higher TP in quantized configs (Sec. 6.5, not applicable here but confirms TP=2 is architecturally clean). 160GB aggregate VRAM against a ~36GB/GPU weight shard leaves large headroom, so default `--gpu-memory-utilization 0.90` and `--max-num-seqs 16` need no tight-fit tuning the way the single-NVL config does. Do not set `VLLM_USE_V2_MODEL_RUNNER=1` (Sec. 6.5).

**(C) Single H100 SXM, FP8 (continuations phase, carried over from Sec. 3/Sec. 6 decision):**
```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3
```
FP8 checkpoint (37.5GB) leaves most of the 80GB card free for KV cache/state and high `--max-num-seqs` concurrency, and H100's native FP8 tensor cores give the throughput the higher-volume continuations phase needs (Sec. 4), no memory-fit tuning required here, unlike (A).

### Verdict summary

- **H100 NVL 94GB single-card BF16: fits, but tight** (~5-13GB headroom depending on `--max-num-seqs`; use 8, not 16, unless `--gpu-memory-utilization` is raised to 0.95). Stock is thin (2 datacenters, LOW each), treat as opportunistic, not primary.
- **2xH100 SXM TP=2 BF16: the recommended primary path**, comfortable memory headroom, clean TP=2 shard on every head count in this architecture, better underlying GPU stock (Sec. 3), and no open vLLM issues found that block plain-BF16 TP=2 for this model (the two live GDN+TP bugs found are either quantization-only and fixed, or gated behind an opt-in flag you simply don't set).
- Three concrete serve commands above cover NVL-BF16, TP2-BF16, and FP8, all three ready to use as-is.
