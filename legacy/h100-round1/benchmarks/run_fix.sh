#!/bin/bash
set -u
until grep -q "REM DONE" ~/run_rem.out 2>/dev/null; do sleep 20; done
sleep 5
sudo docker rm -f gr >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done
DR31="{\"model\":\"google/gemma-4-31B-it-assistant\",\"num_speculative_tokens\":4}"
echo "[fix] launching 31B-bf16-mtp retry util=0.88 maxlen=8192 $(date)"
sudo docker run -d --name gr --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:nightly --model google/gemma-4-31B-it --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --no-enable-prefix-caching --host 0.0.0.0 --port 8000 --speculative-config "$DR31" >/dev/null 2>&1
HEALTHY=0
for i in $(seq 1 240); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { HEALTHY=1; break; }
  sudo docker ps -a --filter name=gr --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "[31B-bf16-mtp-retry] DIED"; sudo docker logs gr 2>&1|grep -iE "error|memory|less than|not support|valueerror|out of"|tail -3; break; }; sleep 6; done
[ "$HEALTHY" = 1 ] && python3 ~/bench_tp.py "31B-bf16-mtp-retry" || echo "[31B-bf16-mtp-retry] FAILED-OR-DIED"
sudo docker rm -f gr >/dev/null 2>&1
echo "######## FIX DONE ########"
