# V2 brief — writeup reframe + figure regeneration

*Work order for an agent. Written 2026-08-31. TK reviews everything before publish; when a decision is marked [TK] leave a note instead of deciding.*

## Goal

Two deliverables, in priority order:

1. **Regenerate all figures** to publication grade using SciencePlots and the orx-figures discipline, keeping this project's semantic palette.
2. **Revise the writeup** (`README.md` is the source of truth — the LW draft is behind it) from findings-first to **method-first**: the method is "resampling under flipped conditions" — counterfactual forking — demonstrated on Value Leakage. Integrate the Bigelow line of work.

Do not run new sampling. Every analysis in v2 must come from existing `artifacts/`. New-sampling ideas go into a "future work" note, not into jobs (§6).

## Repo map (verified)

- `README.md` — current writeup text. Edit this.
- `docs/FINDINGS.md` — factual register, maintained by Claude. Append, never rewrite history.
- `figures/style.py` — bespoke matplotlib style. Palette is semantic: `STEERED "#b5552d"` (sienna) ONLY for swapped/influenced arm; `NEUTRAL "#6f7f92"` (blue-grey) for no-bet; `INK "#2b3a4f"`; `FAINT` for spaghetti. White ground (TK decision 2026-08-20, keep).
- `figures/f2_curves.py … f6_intent_strips.py` — one script per figure. F2 reads `artifacts/step3_summaries.json`; posteriors come from `figures/export_truth.py` → `figure_truth.npz` (reads `artifacts/step3_idata_*.nc` via `swap/model.py`).
- `figures/html/F1_method.html` + `render.sh` — F1 is an HTML schematic rendered by headless Chrome at 3x. Different pipeline; touch last.
- `figures/figure-texts.md` — every string on every figure, canonical. Update it in the same commit as any figure change.
- Envs: figure scripts run under `../external/value_leakage/.venv/bin/python` (plotting) or `../.venv-model/bin/python` (PyMC/ArviZ for export_truth). Do not install into either without checking imports first; prefer `uv run --with scienceplots --with matplotlib …` for the new styling if the venvs lack it.
- `external/refs/forking-fast-2608.19611.pdf` — Bigelow et al. 2026, copied in for this task.
- Outputs: `figures/out/png/` + `figures/out/svg/` (F1–F6). Add `figures/out/pdf/` for vector-first exports.

## Part A — figures

### Tooling

- `pip/uv: SciencePlots` (github.com/garrettj403/SciencePlots). Use as base layer: `plt.style.use(['science', 'no-latex'])` — **`no-latex` mandatory** (don't add a TeX dependency to this repo). Then apply our overrides on top.
- orx-figures skill (github.com/alphaXiv/openresearch-cli, `agent-skills/orx-figures/SKILL.md`) — adopt its discipline, not its style module:
  - Build at final printed size; never build big and shrink. Target: LW post column ≈ 6.5 in wide max; single-panel default 5.5×3.4 in.
  - Vector out: SVG for the post, PDF alongside, PNG at 2x only as fallback for LW upload.
  - Every number from artifacts — no remembered or hardcoded values. If a script has an inline constant that should come from data (grep for `0.62`, `0.26`, `87`), replace with a load or assert it against the loaded value.
  - Uncertainty always visible; caption states what the band is (IQR vs 94% HDI — F2 currently mixes language).
  - Colorblind/greyscale-safe: our sienna/blue-grey/ink survives; verify with a greyscale render of each figure once.
- **Precedence:** TK's semantic palette and white ground BEAT SciencePlots defaults. SciencePlots supplies ticks/spines/sizing/legend defaults only. Implement as: `style.use()` refactored to `plt.style.use(['science','no-latex'])` + our rcParams override dict. Keep `style.py`'s public API (`use()`, `plate()`, `annotate()`) so f2–f6 scripts need minimal edits.

### Per figure

| Fig | Script | Change |
|---|---|---|
| F1 method schematic | `html/F1_method.html` | Keep HTML pipeline. Only sync any wording changes from Part B (e.g. dep formula in caption). Re-render at 3x |
| F2 dep(t) curves | `f2_curves.py` | Restyle. Also: PchipInterpolator through 5 grid points can overshoot below 0 — clip interpolation to [0,1] or draw straight segments between grid points. Keep sienna=still-reversible group per style.py semantics [check: current F2 uses sienna for "still reversible", which is consistent with "influenced"] |
| F3 three-arm | `f3_three_arm.py` | Restyle only |
| F4 denial | `f4_denial.py` | Restyle. The right panel (Denies − Admits) must show the HDI explicitly with the −0.06 [−0.09, −0.03] number on-figure |
| F5 timing | `f5_timing.py` | Restyle. This carries the 81% claim — lock position is a posterior median on a 5-point grid; caption must say so (localization is coarse, ±0.2 in t) |
| F6 intent strips | `f6_intent_strips.py` | Restyle only |

Definition of done per figure: renders from its script with zero warnings, SVG+PDF+PNG written, strings match `figure-texts.md`, greyscale check passed, no `ax.set_title` removed without moving its content to caption text in `figure-texts.md` (blogpost figures MAY keep short lowercase bold titles — that's the existing house style; orx's no-title rule is for papers, don't apply it blindly).

## Part B — writeup reframe

### New spine

Method-first, findings as demonstration. Working title direction: the method is *resampling under flipped conditions* / counterfactual forking. Positioning sentence to include early (TK's wording preference, adapt): **"Forking Paths Analysis asks how undecided the model is; this asks what the decision still depends on."** FPA is observational (same prompt, model's own uncertainty); ours is interventional (do(flip the bet), measure surviving dependence).

Structure target:
1. TL;DR (keep findings bullets — they are the hook)
2. The method, named, with formal definitions (see fixes below)
3. Lineage: FPA → Thought Branches → this (one paragraph)
4. Demonstration: Value Leakage context (compress current Context section ~30%)
5. Results (as-is, restyled figures)
6. What the labels enable (promote from "Why this matters": per-rollout ground truth from an *unmodified* model, vs model organisms — this is the product; sell the labels, not the resampling)
7. Limitations (with changes below)

### Citations to add

- **Bigelow, Holtzman, Tanaka, Ullman, "Forking Paths in Neural Text Generation", arXiv 2412.07961** — the missing ancestor. Outcome-distribution-vs-prefix-position is their o_t; our dep(t) is the distance between two o_t curves under different prompts. Cite where Thought Branches is currently introduced.
- **Bigelow et al., "Forking Fast", arXiv 2608.19611** (PDF in `external/refs/`) — three uses:
  1. Noise model: they show counts at a cut are exactly multinomial (measured −0.49 log-log slope vs −0.5 predicted). This is the formal justification for our Beta-Bernoulli/PyMC treatment — cite in the setup/stats paragraph.
  2. Their smoothing gives 3–5× effective sample size at our S range → cite in the "25 samples can't resolve <0.3" limitation as the known fix.
  3. Their "flat with sparse sharp forks" finding → motivates the dense-cuts future experiment (below).

### Errors to fix in the text (verified against the current draft)

1. **dep prose definition is wrong even though the numbers are right.** "If the original and swapped continuations land on the good side equally often, the bet no longer matters" — under the swap the good side FLIPS, so equal good-side rates would mean p=0.5, not independence. Correct quantity: dep(t) = P(above | original) − P(above | swapped) for an above-good conversation (sign toward the original's good side). At t=0 this equals p_above-good + p_below-good − 1 = the paper's bias — which is why dep(0)=0.62 checks out. Write the formula explicitly; kill the ambiguous sentence.
2. **The ×2 scale conversion is unexplained.** "Mean s(0.2) is +0.27, which on their scale is 0.54" — because s is a one-direction deviation from 0.5 and the paper's bias sums two directions. One sentence.
3. **Limitation 3 (swapped prompt conflicts with prefix) promote to its own paragraph.** It is the strongest objection to "locked": dep→0 could mean "prefix dominates any conflicting prompt", not "decision made". Note `artifacts/splice_check*.jsonl` exists — check with TK [TK] whether splice results are usable before claiming them.
4. Typos in the LW draft only (README already fixed): "p(good **size**…)" in the bias formula; "I'll truncated chains prefixes"; "psychophant" appears nowhere in the post, ignore.
5. F2 caption/legend consistency: bands are IQR in the figure but posterior language elsewhere says HDI — pick one per figure and say which.

### Future-work section (write, don't run)

- Dense cuts in t ∈ [0, 0.3]: distinguishes "commitment is one forking sentence" (Bigelow's sharp forks) from gradual drift. If 87%-by-0.2 becomes ~87%-by-0.05, the claim upgrades to "it's in the framing; the CoT was never in the loop." Requires new sampling — cost estimate ~2 extra cuts × 250 conv × 3 prompts × 25 = 37.5k continuations.
- PELT change-point detection on dep(t) as the lock estimator (Forking Fast's machinery) instead of "dep reaches zero" on a 5-point grid.
- Their smoothing to shrink HDIs at fixed budget, or same precision at S≈10.
- Monitor-reads-first-fifth experiment (already sketched in "Why this matters" — keep).

### Voice constraints

TK's prose: first person, plain, no filler, dry asides allowed ("I believe in posterior supremacy" stays). Do not add em-dash-heavy AI cadence. Do not inflate claims: every number keeps its uncertainty. Spanish-speaker English — do not "correct" idiom into blandness; fix only actual grammar errors.

## Order of work

1. SciencePlots layering into `style.py` (keep API) → regenerate F2–F6 → greyscale check → update `figure-texts.md`.
2. README restructure per §B.
3. F1 HTML wording sync + re-render.
4. Append a dated summary of what changed to `docs/FINDINGS.md`.

## Done criteria

- `for f in f2 f3 f4 f5 f6; do <plot-env> figures/f${f}_*.py; done` exits clean; `figures/out/{svg,pdf,png}/` fully repopulated.
- README: dep formula present; ×2 explained; method-first structure; both Bigelow citations in; limitation-3 paragraph; no new sampling claimed.
- Nothing in `artifacts/` modified. Git status shows only: `figures/*`, `README.md`, `docs/FINDINGS.md`, `docs/V2-BRIEF.md`, `external/refs/`.
