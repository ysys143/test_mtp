#!/bin/bash
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
NI=vllm/vllm-openai:nightly; SI=vllm/vllm-openai:v0.22.1
probe(){ local MODEL=$1 IMG=$2 QUANT=$3 KVD=$4 LABEL=$5
  sudo docker rm -f gp >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done
  ARGS=(--model "$MODEL" --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.95 --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ "$KVD" != auto ] && ARGS+=(--kv-cache-dtype "$KVD")
  [ -n "$QUANT" ] && ARGS+=($QUANT)
  sudo docker run -d --name gp --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -p 8000:8000 "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  LASTRES=TIMEOUT
  for i in $(seq 1 100); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { LASTRES=FIT_262144; break; }
    sudo docker ps -a --filter name=gp --format "{{.Status}}"|grep -qiE "exited|dead" && { LASTRES=OOM; break; }; sleep 6; done
  local KVTOK=$(sudo docker logs gp 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | grep -oE "[0-9,]+" | head -1)
  local ESTMAX=$(sudo docker logs gp 2>&1 | grep -oE "estimated maximum model length is [0-9]+" | tail -1 | grep -oE "[0-9]+")
  echo "CELL|$LABEL|kv=$KVD|$LASTRES|kvtokens=$KVTOK|estmax=$ESTMAX"
  sudo docker rm -f gp >/dev/null 2>&1
}
CELLS=(
 "google/gemma-4-26B-A4B-it|$NI||26B-bf16"
 "google/gemma-4-26B-A4B-it|$NI|--quantization fp8|26B-fp8"
 "google/gemma-4-26B-A4B-it|$NI|--quantization int8_per_channel_weight_only|26B-int8"
 "google/gemma-4-31B-it|$NI||31B-bf16"
 "google/gemma-4-31B-it|$NI|--quantization fp8|31B-fp8"
 "google/gemma-4-31B-it|$NI|--quantization int8_per_channel_weight_only|31B-int8"
 "google/gemma-4-31B-it-qat-w4a16-ct|$NI||31B-qat"
 "google/gemma-4-12B-it|$NI||12B-bf16"
 "google/gemma-4-12B-it|$NI|--quantization fp8|12B-fp8"
 "google/gemma-4-12B-it|$NI|--quantization int8_per_channel_weight_only|12B-int8"
 "Qwen/Qwen3.6-35B-A3B-FP8|$NI||qwen35-fp8"
 "Qwen/Qwen3.6-27B-FP8|$SI||qwen27-fp8"
)
for c in "${CELLS[@]}"; do
  IFS="|" read -r MODEL IMG QUANT LABEL <<< "$c"
  probe "$MODEL" "$IMG" "$QUANT" auto "$LABEL"
  [ "$LASTRES" != FIT_262144 ] && probe "$MODEL" "$IMG" "$QUANT" fp8 "$LABEL"
done
echo "######## SWEEP DONE ########"
