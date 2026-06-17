#!/bin/bash
# think 선검증: Gemma 26B-fp8, reasoning-parser gemma4, enable_thinking=True, 관대한 32K 생성.
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2); export HF_TOKEN=$HF
sudo docker rm -f gd >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
# max-model-len 40960: 프롬프트(~2K) + 생성 32K 수용(잘림 방지). 16K면 생성이 잘려 think 의미 없음.
sudo docker run -d --name gd --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:nightly --model google/gemma-4-26B-A4B-it --quantization fp8 \
  --max-model-len 40960 --gpu-memory-utilization 0.92 --reasoning-parser gemma4 \
  --host 0.0.0.0 --port 8000 >/dev/null 2>&1
H=0; for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }; sudo docker ps -a --filter name=gd --format "{{.Status}}"|grep -qiE "exited|dead" && { echo SERVE-DIED; sudo docker logs gd 2>&1|tail -3; exit; }; sleep 6; done
[ "$H" != 1 ] && { echo NOHEALTH; exit; }
pkill -f inject_proxy.py 2>/dev/null; sleep 1
ENABLE_THINK=true setsid python3 ~/inject_proxy.py >/dev/null 2>&1 & sleep 3
echo "=== Gemma26-fp8 THINK (enable_thinking=true, max_gen_toks=32768, temp0.6/top_p0.95) GPQA --limit 20 ==="
~/lmeval-venv/bin/lm_eval --model local-chat-completions \
  --model_args model=google/gemma-4-26B-A4B-it,base_url=http://localhost:8001/v1/chat/completions,num_concurrent=16,tokenized_requests=False,timeout=3600 \
  --tasks gpqa_diamond_cot_zeroshot --limit 20 --apply_chat_template \
  --gen_kwargs max_gen_toks=32768,temperature=0.6,top_p=0.95 2>&1 | grep -iE "flexible-extract|^\|gpqa|Error:" | tail -4
pkill -f inject_proxy.py 2>/dev/null
sudo docker rm -f gd >/dev/null 2>&1
echo THINK_VERIFY_DONE
