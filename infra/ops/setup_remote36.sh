#!/bin/bash
set -ex
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -U pip
/workspace/venv/bin/pip install "vllm==0.27.1" "huggingface_hub[cli]" hf_transfer
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
/workspace/venv/bin/hf download Qwen/Qwen3.6-35B-A3B-FP8
cat > /workspace/serve.sh << 'EOS'
#!/bin/bash
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/hf
exec /workspace/venv/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 32768 \
  --trust-remote-code \
  --port 8000
EOS
chmod +x /workspace/serve.sh
nohup /workspace/serve.sh > /workspace/vllm.log 2>&1 &
echo SETUP_DONE
