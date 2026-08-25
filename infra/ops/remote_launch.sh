#!/bin/bash
# $1 host $2 ssh-port $3 local requests file $4 model grep
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -p $2 root@$1"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -P $2"
D=/Users/tk/Documents/Personal/ais/projects26/leakage-probing
$SSH 'mkdir -p /workspace/run && rm -f /workspace/run/DONE /workspace/run/results.jsonl' >/dev/null 2>&1
$SCP $D/swap/driver.py $D/swap/template.py $D/swap/cuts.py $D/status/remote_runner.sh root@$1:/workspace/run/ >/dev/null 2>&1
$SCP $3 root@$1:/workspace/run/requests.jsonl >/dev/null 2>&1
$SSH "/workspace/venv/bin/pip install -q httpx >/dev/null 2>&1; chmod +x /workspace/run/remote_runner.sh; nohup /workspace/run/remote_runner.sh '$4' > /workspace/run/runner.log 2>&1 < /dev/null & echo STARTED" 2>/dev/null
