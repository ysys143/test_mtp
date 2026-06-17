#!/bin/bash
set -u
clean_gpu(){ sudo docker rm -f ga >/dev/null 2>&1; sudo pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' '); [ "$u" -lt 2000 ] && return; sleep 3; done; }
accphase(){ local LABEL=$1 IMG=$2 MODEL=$3 EXTRA=${4:-}
  clean_gpu
  ARGS=(--model "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name ga --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
    "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  for i in $(seq 1 200); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    sudo docker ps -a --filter name=ga --format '{{.Status}}'|grep -qiE 'exited|dead' && { echo "[$LABEL] DIED"; return; }; sleep 6; done
  python3 ~/bench_acc.py "$LABEL" 30 2>&1
  sudo docker rm -f ga >/dev/null 2>&1
}
NI=vllm/vllm-openai:nightly; GI=vllm/vllm-openai:gemma
A=google/gemma-4-26B-A4B-it; D=google/diffusiongemma-26B-A4B-it
accphase "acc-26B-bf16" "$NI" "$A" ""
accphase "acc-26B-fp8"  "$NI" "$A" "--quantization fp8"
accphase "acc-26B-int8" "$NI" "$A" "--quantization int8_per_channel_weight_only"
accphase "acc-diff-bf16" "$GI" "$D" "--max-num-seqs 4"
accphase "acc-diff-fp8"  "$GI" "$D" "--max-num-seqs 4 --quantization fp8"
accphase "acc-qwen35-fp8" "$NI" "Qwen/Qwen3.6-35B-A3B-FP8" ""
echo "######## ACC DONE ########"
