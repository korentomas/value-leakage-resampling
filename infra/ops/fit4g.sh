#!/bin/bash
# Qwen3.6 (4G): aggregate judged results -> counts; dep fit (orig/swap + cached t=0 cells); three-arm r fit with plug-in s.
set -e
R=/Users/tk/Documents/Personal/ais/projects26/leakage-probing; cd $R/swap
PY=../external/value_leakage/.venv/bin/python; PM=../.venv-model/bin/python
cat $R/artifacts/step4g_judge_results.jsonl $R/artifacts/step4g_judge_results_w2.jsonl 2>/dev/null > $R/artifacts/step4g_judge_results_all.jsonl
$PY driver.py aggregate --results $R/artifacts/step4g_results_merged.jsonl --judge-results $R/artifacts/step4g_judge_results_all.jsonl --out $R/artifacts/step4g_counts_all.jsonl
$PY - <<'PYEOF'
import json
R='/Users/tk/Documents/Personal/ais/projects26/leakage-probing/artifacts'
rows=[json.loads(l) for l in open(f'{R}/step4g_counts_all.jsonl')]
t0=[json.loads(l) for l in open(f'{R}/step4g_t0_counts.jsonl')]
with open(f'{R}/step4g_counts_2arm.jsonl','w') as f:
    for r in rows:
        if r['arm'] in ('orig','swap'): f.write(json.dumps(r)+'\n')
    for r in t0: f.write(json.dumps(r)+'\n')
with open(f'{R}/step4g_counts_neutral.jsonl','w') as f:
    for r in rows:
        if r['arm']=='neutral': f.write(json.dumps(r)+'\n')
import collections; c=collections.Counter(r['arm'] for r in rows); print('4G counts by arm', dict(c), '+ t0 cells', len(t0))
PYEOF
$PM model.py --counts $R/artifacts/step4g_counts_2arm.jsonl --prior all --informed-dep0 0.27 --seed 0 --out $R/artifacts/step4g_summaries.json --netcdf $R/artifacts/step4g_idata.nc > $R/status/logs/step4g-fit-dep.log 2>&1
mkdir -p $R/artifacts/step4g_fits
$PM three_arm.py --step3-counts $R/artifacts/step4g_counts_2arm.jsonl --neutral-counts $R/artifacts/step4g_counts_neutral.jsonl --baseline-counts $R/artifacts/step4g_baseline_counts.jsonl --step3-summaries $R/artifacts/step4g_summaries.json --pairs r --prior all --informed-dep0 "r=0.135,c=0.135,s=0" --out-dir $R/artifacts/step4g_fits --tag step4g > $R/status/logs/step4g-fit-r.log 2>&1
echo FIT4G_DONE
