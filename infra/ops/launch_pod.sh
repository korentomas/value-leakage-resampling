#!/bin/bash
# $1 name $2 host $3 ssh-port $4 local-port $5 setup-script $6 model-grep ; sets up + keeps a tunnel; no driver
name=$1; host=$2; sport=$3; lport=$4; setup=$5; mg=$6
D=/Users/tk/Documents/Personal/ais/projects26/leakage-probing/status
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -p $sport root@$host"
n=0; until $SSH 'echo ok' >/dev/null 2>&1; do n=$((n+1)); [ $n -gt 40 ] && echo "$name SSH_TIMEOUT" && exit 1; sleep 15; done
$SSH 'bash -s' < $D/$setup > $D/logs/setup-$name.log 2>&1
grep -q SETUP_DONE $D/logs/setup-$name.log || { echo "$name SETUP_FAILED"; exit 1; }
nohup $D/tunnel_keeper.sh $host $sport $lport > $D/logs/tunnel-keeper-$name.log 2>&1 < /dev/null &
n=0; until curl -s -m 5 http://localhost:$lport/v1/models 2>/dev/null | grep -q "$mg"; do n=$((n+1)); [ $n -gt 150 ] && echo "$name VLLM_TIMEOUT" && exit 1; sleep 10; done
echo "$name VLLM_READY $(date -u +%H:%M) localhost:$lport"
