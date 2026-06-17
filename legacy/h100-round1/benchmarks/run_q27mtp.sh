#!/bin/bash
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
sudo docker rm -f gq >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
printf "%s" "{\"method\":\"mtp\",\"num_speculative_tokens\":2}" > ~/qspec.json
sudo docker run -d --name gq --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -p 8000:8000 vllm/vllm-openai:v0.22.1 --model Qwen/Qwen3.6-27B-FP8 --max-model-len 16384 --max-num-seqs 256 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --speculative-config "$(cat ~/qspec.json)" --host 0.0.0.0 --port 8000 >/dev/null 2>&1
H=0; for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }
  sudo docker ps -a --filter name=gq --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "DIED"; sudo docker logs gq 2>&1|grep -iE "error|raise|valueerror"|tail -3; break; }; sleep 6; done
echo "health=$H after $((i*6))s"
[ "$H" = 1 ] && python3 ~/bench_tp.py 27B-mtp || echo "[27B-mtp] STILL-NOT-READY"
sudo docker rm -f gq >/dev/null 2>&1
echo "######## Q27MTP DONE ########"
