#!/bin/bash
# Runs ON the pod: wait for local vLLM ($1 = model grep), run driver on requests.jsonl -> results.jsonl, write DONE
cd /workspace/run
n=0; until curl -s -m 5 http://localhost:8000/v1/models | grep -q "$1"; do n=$((n+1)); [ $n -gt 400 ] && { echo VLLM_TIMEOUT > DONE; exit 1; }; sleep 15; done
/workspace/venv/bin/python driver.py run --requests requests.jsonl --results results.jsonl --base-url http://localhost:8000 --concurrency 64 --timeout 3600 > run.log 2>&1
echo "EXIT=$?" > DONE
