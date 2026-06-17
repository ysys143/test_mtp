#!/bin/bash
set -u
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
SI=vllm/vllm-openai:v0.22.1
clean(){ sudo docker rm -f gq >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d " "); [ "${u:-9999}" -lt 2000 ] && break; sleep 3; done; }
serve(){ local SPEC=$1; clean
  ARGS=(--model Qwen/Qwen3.6-27B-FP8 --max-model-len 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --host 0.0.0.0 --port 8000)
  [ -n "$SPEC" ] && { printf "%s" "$SPEC" > ~/qspec.json; ARGS+=(--speculative-config "$(cat ~/qspec.json)"); }
  sudo docker run -d --name gq --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -p 8000:8000 "$SI" "${ARGS[@]}" >/dev/null 2>&1
  H=0; for i in $(seq 1 60); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; return; }
    sudo docker ps -a --filter name=gq --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "  DIED"; sudo docker logs gq 2>&1|grep -iE "valueerror|error:|not support|raise " | tail -3; return; }; sleep 6; done; }
echo "### Qwen27 base (v0.22.1, max-num-seqs 256)"
serve ""
if [ "$H" = 1 ]; then python3 ~/bench_tp.py 27B-base; python3 ~/bench_acc2.py acc-27B-base 30; else echo "[27B-base] FAILED"; fi
echo "### Qwen27 MTP (embedded mtp, gamma2)"
serve "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"
if [ "$H" = 1 ]; then python3 ~/bench_tp.py 27B-mtp; else echo "[27B-mtp] FAILED"; fi
clean
echo "######## QWEN27B DONE ########"
