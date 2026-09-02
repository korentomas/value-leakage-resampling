#!/bin/bash
# Wait until merged 4G coverage hits 500, then judge wave 2, then fit.
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R/swap; mkdir -p $R/infra/ops/logs
PY=../external/value_leakage/.venv/bin/python
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in the environment}"
for i in $(seq 1 90); do
  n=$($PY - <<'PYEOF'
import json,glob
from driver import iter_jsonl; from pathlib import Path
req={r['req_id'] for r in iter_jsonl(Path('../artifacts/step4g_requests.jsonl'))}
seen={}
for f in sorted(glob.glob('../artifacts/step4g_results_*.jsonl')):
    if 'merged' in f: continue
    for rec in iter_jsonl(Path(f)):
        if rec.get('error') or rec['req_id'] not in req: continue
        seen.setdefault(rec['req_id'],rec)
with open('../artifacts/step4g_results_merged.jsonl','w') as f:
    for r in seen.values(): f.write(json.dumps(r)+'\n')
print(len(seen))
PYEOF
)
  echo "$(date -u +%H:%M) merged $n/500"
  [ "$n" -ge 500 ] && break
  sleep 60
done
# wait for judge wave 1 to finish
while pgrep -f "step4g_judge_requests_w1" >/dev/null; do sleep 30; done
$PY driver.py build-judge-requests --results ../artifacts/step4g_results_merged.jsonl --out ../artifacts/step4g_judge_requests_all.jsonl --judge-model anthropic/claude-sonnet-4.6
$PY - <<'PYEOF'
import json
from driver import iter_jsonl; from pathlib import Path
w1={r['req_id'] for r in iter_jsonl(Path('../artifacts/step4g_judge_requests_w1.jsonl'))}
w2=[r for r in iter_jsonl(Path('../artifacts/step4g_judge_requests_all.jsonl')) if r['req_id'] not in w1]
open('../artifacts/step4g_judge_requests_w2.jsonl','w').write(''.join(json.dumps(r)+'\n' for r in w2)); print('w2',len(w2))
PYEOF
$PY driver.py run --requests ../artifacts/step4g_judge_requests_w2.jsonl --results ../artifacts/step4g_judge_results_w2.jsonl --base-url https://openrouter.ai/api --api-key-env OPENROUTER_API_KEY --concurrency 24 --timeout 120 > ../infra/ops/logs/step4g-judge-w2.log 2>&1
echo JUDGE_DONE
$R/infra/ops/fit4g.sh
