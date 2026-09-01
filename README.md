# Value leakage is settled before the CoT discusses it

*A write-up of work in progress. There is a draft preprint, [Counterfactual Resampling to Analyse Model Behaviour](paper.pdf), which is still being polished. Done as my final project for a BlueDot Impact Technical AI Safety sprint, facilitated by BAISH (Buenos Aires AI Safety Hub).*

**TL;DR:** I measured Value Leakage (Betley et al.) per conversation, cutting CoTs at different points and resampling under a prompt changing what serves the model's values.

* Betley et al.'s Value Leakage bias metric is a population average so it can't discern individual conversations where a model's bias influenced its decision. I designed a method to measure the same bias one conversation at a time by cutting the CoT and resampling under a prompt that flips which answer serves the model's values.
* Qwen3.5 decides which side of the threshold it will land on about 20% through its CoT — before writing any estimate. By that point the text carries 88% [80%, 97%] of the bias measured at t = 0, and putting the bet back into the prompt changes almost nothing.
* Denials don't cluster on the influenced traces. CoTs that deny being influenced lean towards the good side slightly less than admitting ones, this could provide evidence against the model selectively covering its tracks.
* Intent statements are done in retrospective. When a CoT contains which side the model means to land on, it mentions it after the answer's already been locked in 81% of the traces (that do state intent). Under a self-report, the model says the bet didn't influence its estimation.
* The result is a per-rollout label (bias) from an unmodified model. This could prove useful for a probe or a monitor eval doing this supervision, where current research uses model organisms with bias trained in.

Links: [Value Leakage paper](https://arxiv.org/abs/2607.14345), [their LW post](https://www.lesswrong.com/posts/hbMw4Yqw6RnFaExDy/value-leakage-an-llm-s-answers-are-silently-shaped-by-its-1), [my code (and posteriors and continuations)](https://github.com/korentomas/value-leakage-resampling)

![giraffes, a rover on the threshold line, a coin and a heart](figures/giraffe.jpg)

*Figure 0. Qwen trying to count the giraffe spots accurately, despite its heart wanting it to make a good donation. Giraffes via nano banana, thanks to all artists whose work was used to train this model.*

## Introduction

Betley et al. describe Value Leakage: a model's answer to a factual question shifts according to what would serve the model's own values, and does so covertly as the CoT mostly doesn't mention this influence.

Their Donation Bet task poses a Fermi problem, with a note that the user has a bet with a friend that if the estimate is above a threshold they send money to a good cause and if it is below, to a bad one. Their counterfactual is the same question, but with the good and bad causes swapped. I'll call the two versions above-good, where the estimate over the threshold is the charitable one, and below-good, where it's the estimate under it. Since scoring only retains the side of the threshold on which the estimate fell, each rollout can be modelled as a Bernoulli trial, and bias is the deviation from fairness. They also read every CoT and label it as admitting, mentioning, or denying that the bet mattered.

![Figure from Betley et al. (2026)](figures/betley-fig1.png)

*Figure from Betley et al. (2026). Left: one estimation question with and without a donation note. Estimates move toward whichever side sends money to a good cause. Middle: its chain of thought claims to ignore that note. Right: a user who knows nothing about these values ends up misinformed.*

Some of the population averages for the models are:

- 0.81 for Claude Opus 4.8,
- 0.62 for Qwen3.5-35B, and
- 0.16 for GPT-5.6

These population averages representing the model bias don't let us know which specific instances the model's bias actively influenced its response (though, statistically, it's all of them). Yet, some of the responses were influenced and the model changed its response and in other cases the model was going to always respond with the same estimate regardless of the bet.

The original authors suggest the model might be denying the influence selectively, hiding it on the conversations where the bet influenced its decision. If this was true then influenced answers should concentrate on answers with denials, and there is no way to check this if we don't know the bias per trace. This same gap blocks us from developing a monitor that reads the CoT for value leakage, and it would also block us from training a probe on the bias and more mech interp methods.

## Methodology

Steps:

* Take a conversation that had the bet
* Truncate its CoT at a fraction *t* of its sentences, call the retained text the prefix
* Continue the prefix under each of three prompts
  * Original: the one that produced the conversation
  * Swapped: exchanges the two causes, so the opposite answer is now the favoured one
  * No-bet: drops the bet altogether

![the method on one conversation](figures/out/png/F1_method.png)

*Figure 1. One conversation, cut at t = 0.21, continued 25 times under each prompt. Each continuation is scored against the model's own no-bet median, drawn as the horizontal line. dep is the original condition minus the swapped one, and s is the no-bet condition minus one half.*

G is the side of the threshold the original prompt favours. The two measurements are:

$$
\begin{aligned}
\mathrm{dep}(t) &= P(G \mid \text{prefix}_t,\ \text{original}) - P(G \mid \text{prefix}_t,\ \text{swapped}) \\
s(t) &= P(G \mid \text{prefix}_t,\ \text{no-bet}) - 0.5
\end{aligned}
$$

dep is the difference between two prompts on the same prefix. When it's zero it means the bet no longer moves the answer, and I call the conversation locked. Where in the CoT that happens is its lock position. At t = 0 there is no prefix and dep(0) is just Betley et al.'s bias.

I need s because if a conversation is locked it doesn't necessarily mean that it's unbiased. If a trace starts with "the good cause needs a high number so I'll aim high", it would have dep = 0 at all cuts, as every continuation would follow the prefix regardless of their prompt. Therefore, I also need the no-bet condition: with no bet in the prompt and no lean in the text, half the continuations land on each side, because the threshold is the model's own median on the task without any bets.

s and Betley et al.'s bias are not on the same scale. Theirs adds the lean from both versions of the question, and s is the lean in one version only, so the quantity to compare against their bias is 2s.

I got the idea from [Thought Branches](https://arxiv.org/abs/2510.27484), which cuts a CoT at a sentence and resamples the continuation to measure how much that sentence causally influences the final answer. It builds on [Forking Paths Analysis](https://arxiv.org/abs/2412.07961) (Bigelow et al.), which instead resamples at token positions and tracks how the distribution over final answers changes throughout the CoT. Both methods vary the CoT while keeping the prompt fixed, whereas I do the opposite, with the CoT prefix staying identical and only the prompt changing. So instead of asking which sentence changes the answer, I'm asking whether the answer still depends on the bet after however many sentences of reasoning.

## The model decides which side it will land on early in the CoT

dep starts at 0.62 and rapidly falls to 0.26 by t = 0.2. Moreover, the median lock position is a quarter of the way through the CoT.

![dep(t) and lock positions](figures/out/paper/F2_dep.png)

*Figure 2. (a) Mean dep(t) over the 250 conversations with the 95% spread across them. (b) Where each conversation locks, as a fraction of its CoT.*

Averaging them out still hides information here so we can look at them disaggregated. Sorting them by where they lock and drawing one row each, most go dark right after t = 0.2 while a minority stay dependent on the bet all the way to the end.

![one row per conversation](figures/out/paper/F2b_perconv.png)

*Figure 3. One row per conversation sorted by lock position, with gradient signifying how much the bet moves that conversation at each of the cuts.*

At t = 0.2,

* The bet stops mattering as a prompt. Continuing a prefix with or without the bet gives a similar rate of the good side.

  ![three prompt conditions](figures/out/paper/F3_arms.png)

  *Figure 4. Once a prefix exists the original and no-bet conditions sit on top of each other, so the prefix carries the bias and the prompt doesn't.*

* The text has already inherited the bias. Mean s(0.2) is +0.27 [0.25, 0.30], and brackets are 95% posterior intervals throughout. Doubled onto Betley et al.'s scale that is 0.55 [0.49, 0.60], against the 0.62 those same rollouts show at t = 0, so 88% of the bias is already in the text. Most of these prefixes don't contain an estimate yet.

* The bet determines the direction of early commitment. Without it 34% of prefixes lean more than 0.2 toward the bet's side and 35% lean that far the other way, while with the bet those become 70% and 7%.

Here is a trace from the crochet subset, cut at t = 0.21, which ends mid-setup without an estimate:

> ... Total: ~5-6 million shawls. Variable 4: Stitches per Shawl. Shawl sizes vary. A small shawl (60x150cm) vs a large wrap (150x200cm). Gauge: HDC (half double crochet). [...] Let's say 2.5 stitches per cm (Gauge). Dimensions: Average 100cm (width) x 150cm (length). Rows: Length / stitch height.

From here 22 of 25 continuations under the original prompt land above the threshold, 8 of 25 under the swapped one, and 19 of 24 with no bet. So this half-finished arithmetic already leans to the good side, and a contradicting prompt can still pull it back. Their judge labels this trace as denying influence, and its one statement of intent, "I should not bias low", arrives at t = 0.65, long after the cut.

## Denials don't cluster on the influenced traces

If the model were hiding its bias selectively, denials would concentrate on the conversations the bet moved. They don't, whether I measure how far each group leans or how many in each group lean at all:

* On average, denying prefixes lean to the good side slightly less than admitting ones, a difference in mean s of -0.06 [-0.09, -0.03].
* Counting how many in each group were shaped by the bet, meaning the prefix already leans past 0.2 or the swapped prompt can still move it, the gap is -0.12 [-0.16, -0.08].

![share by covertness label](figures/out/paper/F4_denial.png)

*Figure 5. Share of each group shaped by the bet, meaning the prefix already leans past 0.2 or the swapped prompt can still move it, by the label Betley et al.'s judge gave the CoT, with 95% intervals. The Denies minus Admits contrast is -0.12 [-0.16, -0.08], with the whole posterior below zero.*

Betley et al. also judge CoTs on whether the model presents signs of eval-awareness, based on a UK AISI rubric that goes from 1 to 7, with 4 or above counting as aware.

* Splitting the traces on eval-awareness doesn't explain the difference away. For the 137 aware ones it's -0.09 [-0.12, -0.06], and for the other 99, -0.03 [-0.07, +0.01].
* Averaging those two groups back together, weighted by size, we get -0.13 [-0.16, -0.09], against -0.12 [-0.16, -0.08] without the split.

Looking closer at how eval-awareness affects behaviour, it comes with less bias in both groups, and the drop is bigger in the denying ones:

* Denying traces go from +0.30 to +0.17 when the model presents eval-awareness.
* Admitting ones from +0.33 to +0.26.

So eval-aware denying traces are the clearest case of Qwen saying it will ignore the bet and then doing it. I didn't plan this split before running it though, and my notes at the time said to treat awareness as a covariate rather than a finding, so I'd hold it loosely.

After seeing this, I ran a self-report test, replaying the conversations and asking the model whether the donation bet had influenced its estimate. It answers yes in 8% of samples and never in the denying conversations, so the self-report largely follows what it had said in its own CoT. The correlation between a conversation's yes-rate and its s(0.2) is -0.07 [-0.22, +0.08].

## Intent statements are done in retrospective

Of the 250 conversations, 147 state at some point which side the model means to land on. Admitting traces do it more often than denying ones, 68% against 44%, which is roughly what those labels mean. When a side is named it's usually the good one: 103 of the 147 point that way, 12 the other way, and in the remaining 32 the direction isn't clear from the sentence.

In those 147, the statement comes after the answer is already locked 81% of the time.

![timing of intent statements](figures/out/paper/F6_timing.png)

*Figure 6. (a) Where the answer locks and where the first stated intent appears, as fractions of the CoT on the same scale as the cuts. (b) The two subtracted within each conversation, which the overlapping distributions in (a) can't show. A positive difference means the intent came after the answer was already settled, which is the case for 81% of the 147.*

Honesty claims are a different thing, and they behave differently. These are sentences like "I must be accurate", which I could locate in 77 of the denying conversations. Their median position is 0.23 against a median lock of 0.26, and only 45% come after the lock, so they arrive with the decision rather than following it.

![timing of honesty claims](figures/out/paper/F7_honesty.png)

*Figure 7. (a) Where the answer locks and where the first honesty claim appears, for the 77 denying conversations in which a claim could be located, as fractions of the CoT. The lock has median 0.26 and the first claim 0.23. (b) The two subtracted within each conversation. A positive difference means the claim came after the answer was already settled, which is the case for 45% of them.*

The traces that make those claims still lean to the good side, mean s +0.23 against 0.00 for prefixes written with no bet, so saying it doesn't make the answer any less biased.

## Qwen3.6 shows the same shape

I ran a smaller version of all this on Qwen3.6-35B, which scores 0.27 in the paper. 100 conversations, cut at t = 0.2 and t = 1.0 only, for 12,500 continuations. dep(0) comes out at 0.27, matching their published number.

* dep falls from 0.27 to 0.10 by t = 0.2 and reaches 0 by the end, so the answer settles early here too.
* Mean s(0.2) is +0.10 [+0.04, +0.16]. Doubled onto Betley et al.'s scale that is 0.20, and their published bias for this model is 0.27, so 74% of it is already in the text at that cut. For Qwen3.5 the same calculation gave 88%.
* Denying traces lean less than admitting ones again, +0.03 against +0.16, a difference of -0.13 [-0.26, +0.02]. Same sign as Qwen3.5's -0.06, but with only 20 admitters the interval includes zero.

One thing does differ. Qwen3.6's bias is almost entirely a below-good effect: at t = 0.2, s is +0.01 in the above-good version of the question and +0.20 in below-good, where Qwen3.5 leans in both. That matches the trajectories in Betley et al.'s Figure 35.

I didn't run the no-bet baseline or the self-report for this model.

## What the labels are for

What comes out of all this is a label per rollout: this conversation was moved by the bet, this one wasn't, and this much of the movement was already in the text at each cut. I think the labels are worth more than any single number above.

They come from an unmodified model, which is the part I care about. Per-rollout ground truth about a bias today mostly comes from model organisms with the bias trained in, and that buys certainty about the label at the price of studying a model built to be biased. Here the label comes from behaviour under a counterfactual prompt, so the model under study is the released one, and the same construction works on any task where the biasing feature can be swapped in the prompt with everything else held fixed.

If we want a probe for value leakage, this is exactly the supervision it needs, and a monitor can be scored against it too. The timing results already say something about what we'd need from such a monitor, because one that waits for the model to confess, or clears a trace because it promised to be honest, would be wrong in a predictable direction. Whether a monitor reading only the first fifth can call the side is untested here, since most of those prefixes contain no estimate and it would be reading framing, with no arithmetic there yet.

## Models and data

Qwen3.5-35B-A3B in FP8 under vLLM, thinking on, temperature 1. I drew 250 conversations from Betley et al.'s cached bet-condition rollouts, cut each at t = 0.2, 0.4, 0.6, 0.8 and 1.0, and continued every cut 25 times under each of the three prompts, for 93,750 continuations. Per-conversation numbers are posteriors from a joint fit in PyMC across cuts and questions, and the fit never sees the admit/deny labels. Brackets are 95% posterior intervals. [Bigelow et al. (2026)](https://arxiv.org/abs/2608.19611) show the counts at a fixed cut are exactly multinomial, so the Beta-Bernoulli treatment is justified.

## Limitations

1. s tells me the direction of an early commitment but can't tell me one happened. With no bet, 34% of prefixes lean past +0.2 and 35% past -0.2 at t = 0.2, so any raw count of conversations the bet shaped inherits this, which is why I report differences between groups.
2. 25 samples per cell can't resolve a difference below about 0.3, so every per-conversation number is a posterior. Bigelow et al.'s smoothing would buy a little over 3x the effective sample size for free and I didn't use it.
3. The swapped prompt conflicts with the prefix, and this is the objection I take most seriously. When dep reaches zero there are two readings: the answer was settled, or the prefix just outweighs any prompt contradicting it. The model often notices the conflict — 16.4% of swapped continuations explicitly re-read or reverse the mapping, against 1.8% under the original prompt, and 28.2% at t = 0.2. So "locked" really means "follows the prefix over a prompt it can tell is contradicting it". What argues against pure prefix-dominance is the no-bet condition, where the prompt doesn't conflict but is simply absent, and there the prefixes still lean by s = +0.27. A prefix that only overrode contradictions wouldn't carry the lean when nothing is being contradicted. To close it properly I'd want a prompt-strength control.
4. The denial result argues against selective concealment, but separating no introspective access from trained covertness needs activation-level work I didn't do.
5. The lock is a posterior median on a 5-point grid, so "locked by 0.2" means the first measured cut and the true commitment could sit anywhere below it.
6. One model family in FP8 on what is probably the paper's most artificial task.
7. A bug found after the runs: the swapped prompt printed the threshold without thousands separators for 7 of 9 questions. On the two clean questions the numbers sit inside the range of the other seven, so I report the runs as they stand. The Qwen3.6 runs have the fix.

## Future work

None of this was run. Cutting densely in t between 0 and 0.3 would tell us whether this is a gradual drift or a single sentence that settles the answer, since Bigelow et al. find outcome distributions stay flat across long stretches before changing sharply at forking points, sometimes after a single token. If 88% by t = 0.2 became 88% by t = 0.05, the claim would go from "settled early" to "settled in the framing, with the CoT never in the loop".

A change-point estimator would place the lock better than I currently do. PELT with an exact multinomial cost, which is what Bigelow et al. use, would give it a proper interval, where I currently read it off whichever grid node crosses a threshold. And the prompt-strength control from limitation 3 is the experiment I'd most want to run.

## Acknowledgements

This project was done as part of a BlueDot Impact Technical AI Safety project sprint, facilitated by BAISH (Buenos Aires AI Safety Hub). Thanks to Tobias Bersia, my main mentor for this project, and to Guido Bergman, Gonzalo Heredia and Nicolás Martorell for their support. Claude was used for proofreading and editing. The ideas, experiments and text are mine.

---

## Reproducing

### Layout

```
screen/     reproduce the paper's Donation Bet bias on a local vLLM serve (gate before spending compute)
swap/       the experiment: cut -> three-prompt continuations -> judge -> counts -> PyMC fit; plus the follow-ups
figures/    scripts for Figures 1-4 (and the two extra figures in figure-texts.md), shared style
artifacts/  derived data the figures and every reported number read from (raw request/result dumps are gitignored)
infra/      RunPod/vLLM serving notes and the shell scripts used to run the fleet
docs/       FINDINGS.md, the running log of what was tried and what came out
external/   (gitignored) clone of TruthfulAI-research/value_leakage; its cache is the source of every prefix
```

### Environment

```bash
git clone https://github.com/TruthfulAI-research/value_leakage external/value_leakage   # + their data submodule
python -m venv .venv-model && .venv-model/bin/pip install numpy pandas pyarrow pymc arviz matplotlib scienceplots openai anthropic httpx
```

`screen/` and `swap/` import the paper's own prompt templates, question list, threshold routine and judge parser from `external/value_leakage`, so nothing here re-implements their evaluation.

### Pipeline, as run

```bash
# 0. Serve Qwen3.5-35B-A3B in FP8 with vLLM (infra/RECON.md), completions endpoint.
# 1. Gate: does the local serve reproduce the paper's 0.62?
cd screen && python run_screen.py --model-key qwen3.5-35 --base-url http://<host>:8000/v1 --out ../artifacts/screen_qwen3.5-35_fp8.jsonl
python judge.py --in ../artifacts/screen_qwen3.5-35_fp8.jsonl --out ../artifacts/screen_qwen3.5-35_fp8.judged.jsonl
python bias.py  --in ../artifacts/screen_qwen3.5-35_fp8.judged.jsonl --model-key qwen3.5-35      # 0.615, gate passed
python master_table.py                                                                          # -> artifacts/qwen3.5-35_master.parquet (+ paper's labels)

# 2. Main run (swap/): 250 sources x 5 cuts x 25 continuations x {orig, swap}, then the no-bet arm on the same prefixes
cd ../swap
python splice_check.py                                    # full-trace continuation lands on the cached side 20/20
python driver.py build-requests --sources ../artifacts/step3_sources.jsonl --out ../artifacts/step3_requests.jsonl --n-cuts 6 --n-continuations 25
python driver.py run --requests ../artifacts/step3_requests.jsonl --results ../artifacts/step3_results.jsonl --base-url http://<host>:8000
python driver.py build-judge-requests --results ../artifacts/step3_results.jsonl --out ../artifacts/step3_judge_requests.jsonl
python driver.py run --requests ../artifacts/step3_judge_requests.jsonl --results ../artifacts/step3_judge_results.jsonl --base-url https://api.anthropic.com --api-key-env ANTHROPIC_API_KEY
python driver.py aggregate --results ../artifacts/step3_results.jsonl --judge-results ../artifacts/step3_judge_results.jsonl --out ../artifacts/step3_counts.jsonl
python neutral_arm.py                                     # no-bet arm requests from the same prefixes -> run/judge/aggregate as above -> step4_neutral_counts.jsonl
python model.py --counts ../artifacts/step3_counts.jsonl --prior all --out ../artifacts/step3_summaries.json   # dep(t) posteriors, three priors
python three_arm.py                                       # s(t) from the (orig, no-bet) fit

# 3. Follow-ups
python baseline_null.py && python null_check.py           # 4D: 250 no-bet prefixes cut at t = 0.2 (the null for s)
python step4_labels.py && python step4_awareness.py       # s by admit/deny label; eval-awareness split
python prefix_direction.py                                # positions of intent statements and honesty claims
python selfreport.py                                      # 4E: post-hoc "did the bet influence you?" x5 per conversation
python first_number.py                                    # 4F: which prefixes already contain an estimate
python qwen36_gen.py && bash ../infra/ops/fit4g.sh && python qwen36_readout.py   # 4G: Qwen3.6 replication
python reviewer_checks.py && python critique_addenda.py   # robustness tables cited in the post

# 4. Figures (from figures/); writes svg + pdf + png + a greyscale proof for each
cd ../figures && sh html/render.sh && for f in f2_curves f3_three_arm f4_denial f5_timing f6_intent_strips; do ../.venv-model/bin/python $f.py; done
```

Tests: `cd swap && python test_cuts.py && python test_template.py && python test_model.py && python synth_test.py`.
