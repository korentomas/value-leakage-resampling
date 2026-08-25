"""4G readout: Qwen3.6 model-free rates by direction and cut, fitted dep summaries, plug-in s(0.2) by label; side by side with Qwen3.5.
Run after status/fit4g.sh: ../.venv-model/bin/python qwen36_readout.py"""
import json, collections, numpy as np
from pathlib import Path
A = Path('../artifacts')
rows = [json.loads(l) for l in open(A/'step4g_counts_all.jsonl')]
base = {r['cell_id']: r['k']/r['n'] for r in map(json.loads, open(A/'step4g_baseline_counts.jsonl'))}
t0 = [json.loads(l) for l in open(A/'step4g_t0_counts.jsonl')]
agg = collections.defaultdict(lambda: [0, 0])
for r in rows + t0:
    g = 0.0 if r.get('t') == 0 or r.get('is_cell_level') else round(r['frac_sentences']*5)/5
    agg[(r['direction'], r['arm'], g)][0] += r['k']; agg[(r['direction'], r['arm'], g)][1] += r['n']
units = collections.defaultdict(int)
for r in rows:
    if r['arm'] == 'neutral': units[(r['direction'], r['cell_id'])] += 1
pb = {}
for d in ('above_good', 'below_good'):
    tot = sum(n for (dd, c), n in units.items() if dd == d); pb[d] = sum(n*base[c] for (dd, c), n in units.items() if dd == d)/tot
print('Qwen3.6 model-free (p_base unit-weighted', {k: round(v, 3) for k, v in pb.items()}, ')')
print('dir   t    orig  neu   swap | s=neu-base r=orig-neu c=neu-swap dep')
for d in ('above_good', 'below_good'):
    for g in (0.0, 0.2, 1.0):
        o = agg[(d, 'orig', g)]; w = agg[(d, 'swap', g)]; n_ = agg[(d, 'neutral', g)]
        o = o[0]/o[1] if o[1] else float('nan'); w = w[0]/w[1] if w[1] else float('nan'); nn = n_[0]/n_[1] if n_[1] else float('nan')
        print(f"{d[:5]} {g:.1f}  {o:.3f} {nn:.3f} {w:.3f} |  {nn-pb[d]:+.3f}    {o-nn:+.3f}     {nn-w:+.3f}    {o-w:+.3f}")
unp = collections.defaultdict(lambda: [0, 0])
for r in rows: unp[r['arm']][0] += r['n_unparsed']; unp[r['arm']][1] += r['n'] + r['n_unparsed']
print('unparsed by arm', {a: round(v[0]/v[1], 3) for a, v in unp.items()})
S = json.load(open(A/'step4g_summaries.json'))
for pr in ('skeptical', 'neutral', 'informed'):
    if pr in S: c = S[pr]['covertness']; print(f"dep fit {pr}: reversible past t0 (P>.9) {c['dependent']}/{c['n']}; conv {S[pr]['convergence']}")
su = S['neutral']['summaries']; units_r = [u for u in su if u['is_rollout']]
dm = np.array([u['dep_mean'] for u in units_r]); print('population dep(t) mean over units (grid idx 0,1,5):', np.round(dm[:, [0, 1, 5]].mean(0), 3))
try:
    R = json.load(open(A/'step4g_fits/step4g_summaries_r.json'))
    for pr in R:
        pt = [u for u in R[pr]['point_s_plugin_from_r'] if u['is_rollout']]; m = np.array([u['mean'] for u in pt])
        print(f"plug-in s(0.2) {pr}: mean {m.mean():+.3f} median {np.median(m):+.3f} share>+0.2 {np.mean(m>0.2):.2f} share<-0.2 {np.mean(m<-0.2):.2f}; conv {R[pr]['convergence']}")
    lab = {}
    import pandas as pd
    df = pd.read_parquet(A/'qwen3.6-35_master.parquet'); lab = dict(zip(df['rollout_id'], df['denial_bucket'])) if 'denial_bucket' in df else {}
    pt = {u['unit_id']: u['mean'] for u in R['neutral']['point_s_plugin_from_r'] if u['is_rollout']}
    byl = collections.defaultdict(list)
    for u, m in pt.items(): byl[lab.get(u)].append(m)
    print('s(0.2) by label:', {k: (len(v), round(float(np.mean(v)), 3)) for k, v in byl.items()})
except Exception as e:
    print('r-fit readout skipped:', e)
