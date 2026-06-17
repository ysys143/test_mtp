#!/bin/bash
set -u
clean_gpu(){ sudo docker rm -f gr >/dev/null 2>&1; sudo pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' '); [ "$u" -lt 2000 ] && return; sleep 3; done; }
serve(){ # IMG MODEL SPEC EXTRA UTIL MAXLEN -> sets HEALTHY=1/0
  local IMG=$1 MODEL=$2 SPEC=$3 EXTRA=${4:-} UTIL=${5:-0.90} MAXLEN=${6:-16384}
  clean_gpu
  ARGS=(--model "$MODEL" --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ -n "$SPEC" ] && { printf '%s' "$SPEC" > ~/rspec.json; ARGS+=(--speculative-config "$(cat ~/rspec.json)"); }
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name gr --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
    "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  HEALTHY=0
  for i in $(seq 1 240); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { HEALTHY=1; return; }
    sudo docker ps -a --filter name=gr --format '{{.Status}}'|grep -qiE 'exited|dead' && { echo "DIED"; sudo docker logs gr 2>&1|grep -iE 'error|less than|not support|valueerror'|tail -2; return; }; sleep 6; done; }
both(){ local LBL=$1; shift; serve "$@"; [ "$HEALTHY" = 1 ] && { python3 ~/bench_tp.py "$LBL"; python3 ~/bench_acc.py "acc-$LBL" 30; } || echo "[$LBL] FAILED"; sudo docker rm -f gr >/dev/null 2>&1; }
tponly(){ local LBL=$1; shift; serve "$@"; [ "$HEALTHY" = 1 ] && python3 ~/bench_tp.py "$LBL" || echo "[$LBL] FAILED"; sudo docker rm -f gr >/dev/null 2>&1; }
NI=vllm/vllm-openai:nightly; GI=vllm/vllm-openai:gemma
D=google/diffusiongemma-26B-A4B-it
# 1) diffusion int8
both "26B-diff-int8" "$GI" "$D" "" "--max-num-seqs 4 --quantization int8_per_channel_weight_only" 0.90 16384
# 2) 31B dense (재다운로드 자동)
G31=google/gemma-4-31B-it; DR31="{\"model\":\"google/gemma-4-31B-it-assistant\",\"num_speculative_tokens\":4}"
both   "31B-bf16-base" "$NI" "$G31" "" "" 0.95 12288
tponly "31B-bf16-mtp"  "$NI" "$G31" "$DR31" "" 0.95 12288
both   "31B-fp8-base"  "$NI" "$G31" "" "--quantization fp8" 0.90 16384
tponly "31B-fp8-mtp"   "$NI" "$G31" "$DR31" "--quantization fp8" 0.90 16384
both   "31B-qat-base"  "$NI" "google/gemma-4-31B-it-qat-w4a16-ct" "" "" 0.90 16384
tponly "31B-qat-mtp"   "$NI" "google/gemma-4-31B-it-qat-w4a16-ct" "{\"model\":\"google/gemma-4-31B-it-qat-q4_0-unquantized-assistant\",\"num_speculative_tokens\":4}" "" 0.90 16384
# 3) 12B dense
G12=google/gemma-4-12B-it; DR12="{\"model\":\"google/gemma-4-12B-it-assistant\",\"num_speculative_tokens\":4}"
both   "12B-bf16-base" "$NI" "$G12" "" "" 0.90 16384
tponly "12B-bf16-mtp"  "$NI" "$G12" "$DR12" "" 0.90 16384
both   "12B-fp8-base"  "$NI" "$G12" "" "--quantization fp8" 0.90 16384
tponly "12B-fp8-mtp"   "$NI" "$G12" "$DR12" "--quantization fp8" 0.90 16384
echo "######## REM DONE ########"
