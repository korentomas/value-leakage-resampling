#!/bin/bash
# $1 name $2 host $3 port $4 setup script $5 requests file $6 model grep : run setup on pod (no tunnel), then start on-pod runner (it waits for vLLM itself)
D=$(cd "$(dirname "$0")" && pwd); mkdir -p $D/logs
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -p $3 root@$2"
n=0; until $SSH 'echo ok' >/dev/null 2>&1; do n=$((n+1)); [ $n -gt 40 ] && echo "$1 SSH_TIMEOUT" && exit 1; sleep 15; done
$SSH 'bash -s' < $D/$4 > $D/logs/setup-$1.log 2>&1
grep -q SETUP_DONE $D/logs/setup-$1.log || { echo "$1 SETUP_FAILED"; exit 1; }
$D/remote_launch.sh $2 $3 $5 $6 && echo "$1 RUNNER_STARTED $(date -u +%H:%M)"
