#!/bin/bash
set -u
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh | head -1 | cut -d= -f2)
sudo docker rm -f gq >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done
echo "[qwen35-acc] serving on nightly $(date)"
sudo docker run -d --name gq --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:nightly --model Qwen/Qwen3.6-35B-A3B-FP8 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --host 0.0.0.0 --port 8000 >/dev/null 2>&1
H=0; for i in $(seq 1 240); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }
  sudo docker ps -a --filter name=gq --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "[qwen35-acc] DIED"; sudo docker logs gq 2>&1|tail -3; break; }; sleep 6; done
if [ "$H" = 1 ]; then
  python3 ~/bench_acc2.py acc-qwen35-fp8 30
  echo "--- 대조(옛 채점기 max_tokens=24) ---"
  python3 ~/bench_acc.py acc-qwen35-fp8-oldscorer 30
fi
sudo docker rm -f gq >/dev/null 2>&1
echo "######## QWEN35ACC DONE ########"
