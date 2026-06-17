#!/bin/bash
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
NI=vllm/vllm-openai:nightly; SI=vllm/vllm-openai:v0.22.1
serve(){ local MODEL=$1 IMG=$2 QUANT=$3 KVD=$4
  sudo docker rm -f gp >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done
  ARGS=(--model "$MODEL" --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.95 --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ "$KVD" != auto ] && ARGS+=(--kv-cache-dtype "$KVD")
  [ -n "$QUANT" ] && ARGS+=($QUANT)
  sudo docker run -d --name gp --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -p 8000:8000 "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  H=0; for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; return; }
    sudo docker ps -a --filter name=gp --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "  SERVE-DIED"; return; }; sleep 6; done; }
run(){ echo "### NIAH $1"; serve "$2" "$3" "$4" "$5"
  [ "$H" = 1 ] && python3 ~/niah.py "$1" 240000 0.1,0.5,0.9 || echo "[$1] SERVE FAILED"
  sudo docker rm -f gp >/dev/null 2>&1; }
run 26B-fp8        google/gemma-4-26B-A4B-it "$NI" "--quantization fp8" auto
run 31B-bf16-kvfp8 google/gemma-4-31B-it     "$NI" "" fp8
run 12B-fp8        google/gemma-4-12B-it     "$NI" "--quantization fp8" auto
run qwen35-fp8     Qwen/Qwen3.6-35B-A3B-FP8  "$NI" "" auto
run qwen27-fp8     Qwen/Qwen3.6-27B-FP8      "$SI" "" auto
echo "######## NIAH VALIDATE DONE ########"
