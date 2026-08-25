"""Split the remaining 4D requests across pods. Usage: python reshard4d.py name:port [name:port ...]
Reads completed req_ids from every artifacts/step4d_results*.jsonl, writes artifacts/step4d_requests_<name>.jsonl
and launches one detached driver per shard writing artifacts/step4d_results_<name>.jsonl."""
import json, sys, glob, subprocess, os
R = '/Users/tk/Documents/Personal/ais/projects26/leakage-probing'
sys.path.insert(0, f'{R}/swap')
from driver import iter_jsonl
from pathlib import Path
done = set()
for f in glob.glob(f'{R}/artifacts/step4d_results*.jsonl'):
    for rec in iter_jsonl(Path(f)):
        if not rec.get('error'): done.add(rec['req_id'])
reqs = [r for r in iter_jsonl(Path(f'{R}/artifacts/step4d_requests.jsonl')) if r['req_id'] not in done]
shards = [a.split(':') for a in sys.argv[1:]]
print(f'done {len(done)}, remaining {len(reqs)}, shards {len(shards)}')
PY = f'{R}/external/value_leakage/.venv/bin/python'
for i, (name, port) in enumerate(shards):
    part = reqs[i::len(shards)]
    p = f'{R}/artifacts/step4d_requests_{name}.jsonl'
    with open(p, 'w') as f:
        for r in part: f.write(json.dumps(r) + '\n')
    cmd = (f'cd {R}/swap && {PY} driver.py run --requests {p} --results {R}/artifacts/step4d_results_{name}.jsonl '
           f'--base-url http://localhost:{port} --concurrency 64 --timeout 3600 > {R}/status/logs/step4d-run-{name}.log 2>&1; '
           f'echo "4D_EXIT={name}=$?" >> {R}/status/logs/step4d-run-{name}.log')
    subprocess.Popen(['nohup', 'bash', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f'  {name}: {len(part)} requests -> localhost:{port}')
