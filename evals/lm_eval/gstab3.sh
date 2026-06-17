#!/bin/bash
# 어려운 문제로 Gemma(gemma4)·Qwen(qwen3) reasoning-parser + thinking 분리/토큰 확인
HF=$(grep -oE "HF_TOKEN=hf_[A-Za-z0-9]+" ~/run_rem.sh|head -1|cut -d= -f2)
NI=vllm/vllm-openai:nightly
HARD='A particle moves along x with v(t)=3t^2-12t+9 (m/s). Over t in [0,5]s, what is the total distance traveled (not displacement)? Reason carefully, end with: The answer is (X) meters.'
probe(){ # MODEL EXTRA PARSER LABEL THINKARG
  local MODEL=$1 EXTRA=$2 PARSER=$3 LABEL=$4
  sudo docker rm -f gd >/dev/null 2>&1; sudo pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
  ARGS=(--model "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser "$PARSER" --host 0.0.0.0 --port 8000)
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  sudo docker run -d --name gd --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 "$NI" "${ARGS[@]}" >/dev/null 2>&1
  local H=0; for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { H=1; break; }; sudo docker ps -a --filter name=gd --format "{{.Status}}"|grep -qiE "exited|dead" && { echo "[$LABEL] DIED"; sudo docker logs gd 2>&1|grep -iE "parser|invalid|choose from|error"|tail -5; return; }; sleep 6; done
  [ "$H" != 1 ] && { echo "[$LABEL] NOHEALTH"; return; }
  Q="$HARD" LBL="$LABEL" MODEL="$MODEL" python3 - <<'PYEOF'
import urllib.request, json, os
URL="http://localhost:8000/v1/chat/completions"; M=os.environ["MODEL"]; q=os.environ["Q"]; lbl=os.environ["LBL"]
for et in [True, False]:
    body={"model":M,"messages":[{"role":"user","content":q}],"max_tokens":8000,"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":et}}
    r=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        d=json.load(urllib.request.urlopen(r,timeout=600)); c=d["choices"][0]; m=c["message"]
        rc=m.get("reasoning_content") or ""; ct=m.get("content") or ""
        print("[%s] et=%s finish=%s tok=%s reasoning_len=%d content_len=%d"%(lbl,et,c.get("finish_reason"),d["usage"]["completion_tokens"],len(rc),len(ct)))
        print("   content_tail:", repr(ct[-120:]))
    except Exception as e:
        print("[%s] et=%s ERR %s"%(lbl,et,str(e)[:120]))
PYEOF
  sudo docker rm -f gd >/dev/null 2>&1
}
probe "google/gemma-4-26B-A4B-it" "--quantization fp8" gemma4 GEMMA26
probe "Qwen/Qwen3.6-35B-A3B-FP8" "" qwen3 QWEN35
echo GSTAB3_DONE
