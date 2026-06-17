#!/bin/bash
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
export HF_TOKEN=$HF
NI=vllm/vllm-openai:nightly; SI=vllm/vllm-openai:v0.22.1
TASKS=${TASKS:-mmlu_pro}
serve(){ local MODEL=$1 IMG=$2 EXTRA=$3
  sudo docker rm -f gl >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done
  ARGS=(--model "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8000)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name gl --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  H=0; for i in $(seq 1 150); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; return; }; sudo docker ps -a --filter name=gl --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "  SERVE-DIED"; sudo docker logs gl 2>&1|grep -iE "error|memory|oom"|tail -2; return; }; sleep 6; done; }
run(){ local LABEL=$1 MODEL=$2 IMG=$3 EXTRA=$4
  echo "### CELL $LABEL $(date +%H:%M)"; serve "$MODEL" "$IMG" "$EXTRA"
  [ "$H" != 1 ] && { echo "[$LABEL] SERVE FAILED"; return; }
  ~/lmeval-venv/bin/lm_eval --model local-chat-completions \
    --model_args model=$MODEL,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,tokenized_requests=False,timeout=1200 \
    --tasks $TASKS --apply_chat_template --output_path ~/lmeval_results/$LABEL 2>&1 | grep -iE "^\|.*exact_match|^\|mmlu_pro|^\|bbh|Error:|gated" | tail -20
  echo "[$LABEL] DONE $(date +%H:%M)"
  sudo docker rm -f gl >/dev/null 2>&1
}
run 26B-bf16 google/gemma-4-26B-A4B-it "$NI" ""
run 26B-fp8  google/gemma-4-26B-A4B-it "$NI" "--quantization fp8"
run 26B-int8 google/gemma-4-26B-A4B-it "$NI" "--quantization int8_per_channel_weight_only"
run 31B-bf16 google/gemma-4-31B-it     "$NI" ""
run 31B-fp8  google/gemma-4-31B-it     "$NI" "--quantization fp8"
run 31B-qat  google/gemma-4-31B-it-qat-w4a16-ct "$NI" ""
run 12B-bf16 google/gemma-4-12B-it     "$NI" ""
run 12B-fp8  google/gemma-4-12B-it     "$NI" "--quantization fp8"
run qwen35-fp8 Qwen/Qwen3.6-35B-A3B-FP8 "$NI" ""
run qwen27-fp8 Qwen/Qwen3.6-27B-FP8     "$SI" "--max-num-seqs 256"
echo "######## LMEVAL MATRIX DONE ########"
