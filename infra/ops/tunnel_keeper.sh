#!/bin/bash
# Keep an ssh -L tunnel alive: $1 host $2 ssh-port $3 local-port
while true; do
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N -L $3:localhost:8000 -p $2 root@$1
  echo "$(date -u +%H:%M:%S) tunnel exited rc=$?; restarting" 
  sleep 5
done
