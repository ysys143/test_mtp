#!/bin/bash
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2); export HF_TOKEN=$HF
NI=vllm/vllm-openai:nightly; SI=vllm/vllm-openai:v0.22.1
clean(){ sudo docker rm -f gd >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done; }
serve(){ local MODEL=$1 IMG=$2 EXTRA=$3; clean
  ARGS=(--model "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8000)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name gd --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  H=0; for i in $(seq 1 150); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; return; }; sudo docker ps -a --filter name=gd --format "{{.Status}}"|grep -qiE "exited|dead" && { echo SERVE-DIED; return; }; sleep 6; done; }
evalcell(){ local LABEL=$1 MODEL=$2 IMG=$3 EXTRA=$4 TASK=$5 PROXY=$6
  echo "### $LABEL [$TASK] proxy=$PROXY $(date +%H:%M)"; serve "$MODEL" "$IMG" "$EXTRA"
  [ "$H" != 1 ] && { echo "[$LABEL/$TASK] SERVE FAILED"; return; }
  local BASE=http://localhost:8000/v1/chat/completions
  if [ "$PROXY" = 1 ]; then pkill -f nothink_proxy.py 2>/dev/null; sleep 1; setsid python3 ~/nothink_proxy.py >/dev/null 2>&1 & sleep 3; BASE=http://localhost:8001/v1/chat/completions; fi
  ~/lmeval-venv/bin/lm_eval --model local-chat-completions --model_args model=$MODEL,base_url=$BASE,num_concurrent=48,tokenized_requests=False,timeout=1800 --tasks $TASK --apply_chat_template --output_path ~/lmeval_results/$LABEL 2>&1 | grep -iE "^\|.*exact_match|^\|gpqa|^\|mmlu_pro|Error:" | tail -8
  [ "$PROXY" = 1 ] && pkill -f nothink_proxy.py 2>/dev/null
  echo "[$LABEL/$TASK] DONE $(date +%H:%M)"; sudo docker rm -f gd >/dev/null 2>&1
}
# Qwen MMLU-Pro (proxy, no-think)
evalcell qwen35-fp8 Qwen/Qwen3.6-35B-A3B-FP8 "$NI" "" mmlu_pro 1
evalcell qwen27-fp8 Qwen/Qwen3.6-27B-FP8 "$SI" "--max-num-seqs 256" mmlu_pro 1
# GPQA-Diamond all cells (Gemma direct, Qwen proxy)
evalcell 26B-bf16 google/gemma-4-26B-A4B-it "$NI" "" gpqa_diamond_cot_zeroshot 0
evalcell 26B-fp8  google/gemma-4-26B-A4B-it "$NI" "--quantization fp8" gpqa_diamond_cot_zeroshot 0
evalcell 26B-int8 google/gemma-4-26B-A4B-it "$NI" "--quantization int8_per_channel_weight_only" gpqa_diamond_cot_zeroshot 0
evalcell 31B-bf16 google/gemma-4-31B-it "$NI" "" gpqa_diamond_cot_zeroshot 0
evalcell 31B-fp8  google/gemma-4-31B-it "$NI" "--quantization fp8" gpqa_diamond_cot_zeroshot 0
evalcell 31B-qat  google/gemma-4-31B-it-qat-w4a16-ct "$NI" "" gpqa_diamond_cot_zeroshot 0
evalcell 12B-bf16 google/gemma-4-12B-it "$NI" "" gpqa_diamond_cot_zeroshot 0
evalcell 12B-fp8  google/gemma-4-12B-it "$NI" "--quantization fp8" gpqa_diamond_cot_zeroshot 0
evalcell qwen35-fp8 Qwen/Qwen3.6-35B-A3B-FP8 "$NI" "" gpqa_diamond_cot_zeroshot 1
evalcell qwen27-fp8 Qwen/Qwen3.6-27B-FP8 "$SI" "--max-num-seqs 256" gpqa_diamond_cot_zeroshot 1
echo "######## FINAL DONE ########"
