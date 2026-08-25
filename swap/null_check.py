"""4D: null for s. Baseline-sourced prefixes cut at t=0.2, continued under the no-bet prompt.
s_null = k/n - 0.5 per prefix (k = above threshold). Compare |s_null| with the bet arm's s(0.2).
Run: ../.venv-model/bin/python null_check.py  (reads artifacts/step4d_counts.jsonl, step4_fits/neutral/step4_summaries_r.json)"""
import json, numpy as np, sys
from pathlib import Path
A = Path('../artifacts')
rows = [json.loads(l) for l in open(A/'step4d_counts.jsonl')]
rng = np.random.default_rng(0)
k = np.array([r['k'] for r in rows]); n = np.array([r['n'] for r in rows]); unp = np.array([r['n_unparsed'] for r in rows])
s_raw = k/np.maximum(n,1) - 0.5
print(f"prefixes {len(rows)}  parsed per cell mean {n.mean():.1f}  unparsed rate {unp.sum()/(n.sum()+unp.sum()):.3f}")
print(f"raw |s_null|: mean {np.abs(s_raw).mean():.3f}  median {np.median(np.abs(s_raw)):.3f}  share > 0.2: {np.mean(np.abs(s_raw)>0.2):.3f}  share > 0.3: {np.mean(np.abs(s_raw)>0.3):.3f}")
print(f"raw s_null signed: mean {s_raw.mean():+.3f} (should be ~0); share > +0.2: {np.mean(s_raw>0.2):.3f}; share < -0.2: {np.mean(s_raw<-0.2):.3f}")
# Beta posterior per prefix (Jeffreys), propagated fractions, and classified (P>0.9)
draws = rng.beta(k+0.5, n-k+0.5, size=(4000, len(rows))) - 0.5
for eps in (0.1, 0.15, 0.2, 0.3):
    p_abs = (np.abs(draws) > eps).mean(axis=0)
    prop = (np.abs(draws) > eps).mean(axis=1)
    print(f"eps {eps}: propagated share |s_null|>eps {prop.mean():.3f} [{np.percentile(prop,2.5):.3f},{np.percentile(prop,97.5):.3f}]  classified P>0.9: {int((p_abs>0.9).sum())}/{len(rows)}  P(|s|<eps)>0.9: {int(((np.abs(draws)<eps).mean(axis=0)>0.9).sum())}")
# by question
import collections
byq = collections.defaultdict(list)
for r, s in zip(rows, s_raw): byq[r['prompt_key']].append(abs(s))
print("per question mean |s_null|:", {q.replace('v1_','').replace('_accurate',''): round(float(np.mean(v)),3) for q,v in sorted(byq.items())})
# compare with bet-arm s(0.2)
S = json.load(open(A/'step4_fits/neutral/step4_summaries_r.json'))['neutral']
bet = np.array([u['mean'] for u in S['point_s_plugin_from_r'] if u['is_rollout']])
print(f"bet arm s(0.2): mean {bet.mean():+.3f}  median {np.median(bet):+.3f}  share > 0.2: {np.mean(bet>0.2):.3f}  share |s|>0.2: {np.mean(np.abs(bet)>0.2):.3f}")
null_share = np.mean(np.abs(s_raw)>0.2)
print(f"symmetric reading: bet-arm share s>+0.2 ({np.mean(bet>0.2):.3f}) minus null share of |s|>0.2 landing on a random 'good' side ({null_share/2:.3f}) = {np.mean(bet>0.2)-null_share/2:.3f}")
json.dump({'n':len(rows),'raw_abs_mean':float(np.abs(s_raw).mean()),'raw_share_gt02':float(null_share),'signed_mean':float(s_raw.mean()),
           'bet_share_gt02':float(np.mean(bet>0.2)),'bet_mean':float(bet.mean())}, open(A/'step4d_summary.json','w'), indent=1)
