#!/bin/bash
set -u
clean_gpu(){ sudo docker rm -f gq >/dev/null 2>&1; sudo pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' '); [ "$u" -lt 2000 ] && return; sleep 3; done; }
diag(){ local LABEL=$1 MODEL=$2 SPEC=$3 EXTRA=${4:-}
  clean_gpu
  ARGS=(--model "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ -n "$SPEC" ] && { printf '%s' "$SPEC" > ~/qspec.json; ARGS+=(--speculative-config "$(cat ~/qspec.json)"); }
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  echo "===== DIAG $LABEL : $MODEL spec=${SPEC:-none} extra=${EXTRA:-none} ====="
  sudo docker run -d --name gq --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
    vllm/vllm-openai:nightly "${ARGS[@]}" >/dev/null 2>&1
  for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo "$LABEL HEALTHY"; python3 ~/bench_tp.py "$LABEL"; break; }
    sudo docker ps -a --filter name=gq --format '{{.Status}}'|grep -qiE 'exited|dead' && { echo "$LABEL DIED -- ROOT CAUSE:"; sudo docker logs gq 2>&1 | grep -v blob | grep -iE 'error|valueerror|runtimeerror|assert|not.*support|nan|shape|KeyError|nextn|mtp|num_speculative' | tail -8; break; }; sleep 6; done
  sudo docker rm -f gq >/dev/null 2>&1
}
# 27B: base(no spec), mtp(spec), 그리고 enforce-eager 변형
diag "qwen27-base-eager" "Qwen/Qwen3.6-27B-FP8" "" "--enforce-eager"
diag "qwen27-mtp1" "Qwen/Qwen3.6-27B-FP8" "{\"method\":\"mtp\",\"num_speculative_tokens\":1}" ""
diag "qwen35-mtp1" "Qwen/Qwen3.6-35B-A3B-FP8" "{\"method\":\"mtp\",\"num_speculative_tokens\":1}" ""
echo "######## QWEN DIAG DONE ########"
