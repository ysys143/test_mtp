#!/bin/bash
# Gemma4 reasoning-parser 안정화 검증: --reasoning-parser gemma4 + enable_thinking True/False
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
sudo docker rm -f gd >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
sudo docker run -d --name gd --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:nightly --model google/gemma-4-26B-A4B-it --quantization fp8 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser gemma4 \
  --host 0.0.0.0 --port 8000 >/dev/null 2>&1
H=0
for i in $(seq 1 90); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }
  sudo docker ps -a --filter name=gd --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "SERVE-DIED (parser name?)"; sudo docker logs gd 2>&1 | grep -iE "reasoning|parser|invalid|choose from|error" | tail -8; exit; }
  sleep 6
done
[ "$H" != 1 ] && { echo NOHEALTH; exit; }
python3 - <<'PYEOF'
import urllib.request, json
URL="http://localhost:8000/v1/chat/completions"; M="google/gemma-4-26B-A4B-it"
q="A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? End with: The answer is (X)."
for et in [True, False]:
    body={"model":M,"messages":[{"role":"user","content":q}],"max_tokens":4096,"temperature":0,"chat_template_kwargs":{"enable_thinking":et}}
    r=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=300)); c=d["choices"][0]; m=c["message"]
    rc=m.get("reasoning_content"); ct=m.get("content") or ""
    print("=== enable_thinking=%s finish=%s tok=%s ==="%(et,c.get("finish_reason"),d["usage"]["completion_tokens"]))
    print("reasoning_content_len:", len(rc) if rc else 0)
    print("content_repr:", repr(ct[:300]))
PYEOF
sudo docker rm -f gd >/dev/null 2>&1
echo GSTAB2_DONE
