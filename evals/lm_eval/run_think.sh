#!/bin/bash
# think 매트릭스: 전 모델 enable_thinking=True, 관대한 32K(자연 완결), 동일 샘플링. GPQA full.
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2); export HF_TOKEN=$HF
NI=vllm/vllm-openai:nightly; SI=vllm/vllm-openai:v0.22.1; G=gpqa_diamond_cot_zeroshot
clean(){ sudo docker rm -f gd >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done; }
cell(){ local LABEL=$1 MODEL=$2 IMG=$3 EXTRA=$4 PARSER=$5
  echo "### $LABEL [think] $(date +%H:%M)"; clean
  ARGS=(--model "$MODEL" --max-model-len 40960 --gpu-memory-utilization 0.92 --reasoning-parser "$PARSER" --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name gd --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 "$IMG" "${ARGS[@]}" >/dev/null 2>&1
  local H=0; for i in $(seq 1 150); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }; sudo docker ps -a --filter name=gd --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "[$LABEL] SERVE-DIED"; sudo docker logs gd 2>&1|grep -iE "parser|invalid|error|reasoning"|tail -3; return; }; sleep 6; done
  [ "$H" != 1 ] && { echo "[$LABEL] NOHEALTH"; return; }
  pkill -f inject_proxy.py 2>/dev/null; sleep 1; ENABLE_THINK=true setsid python3 ~/inject_proxy.py >/dev/null 2>&1 & sleep 3
  ~/lmeval-venv/bin/lm_eval --model local-chat-completions \
    --model_args model=$MODEL,base_url=http://localhost:8001/v1/chat/completions,num_concurrent=16,tokenized_requests=False,timeout=3600 \
    --tasks $G --apply_chat_template --gen_kwargs max_gen_toks=32768,temperature=0.6,top_p=0.95 \
    --output_path ~/lmeval_results/${LABEL}-think 2>&1 | grep -iE "flexible-extract|^\|gpqa|Error:" | tail -3
  pkill -f inject_proxy.py 2>/dev/null
  echo "[$LABEL] DONE $(date +%H:%M)"; sudo docker rm -f gd >/dev/null 2>&1
}
cell 26B-fp8 google/gemma-4-26B-A4B-it "$NI" "--quantization fp8" gemma4
cell 31B-fp8 google/gemma-4-31B-it     "$NI" "--quantization fp8" gemma4
cell 12B-fp8 google/gemma-4-12B-it     "$NI" "--quantization fp8" gemma4
cell qwen35-fp8 Qwen/Qwen3.6-35B-A3B-FP8 "$NI" "" qwen3
cell qwen27-fp8 Qwen/Qwen3.6-27B-FP8 "$SI" "--max-num-seqs 256" qwen3
echo "######## THINK MATRIX DONE ########"
