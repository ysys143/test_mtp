#!/usr/bin/env python3
# NIAH (needle-in-a-haystack): 긴 컨텍스트 여러 깊이에 사실을 심고 회수율 측정.
# 토큰수는 vLLM /tokenize 엔드포인트(=서버에 떠 있는 그 모델의 실제 토크나이저)로 정확히 측정.
# 사용: python3 niah.py <label> <target_tokens|0=near-max> [depths_csv]
import urllib.request, json, sys
BASE = "http://localhost:8000"
URL = BASE + "/v1/chat/completions"; TOK = BASE + "/tokenize"

def post(url, obj, timeout=600):
    r = urllib.request.Request(url, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)

def detect():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]
M = detect()

CODE = "ZX-4492-QHM"
NEEDLE = f"Special fact to remember: the access code for building 7 is {CODE}. Remember this code."

def filler_line(i):
    return (f"2026-06-14 [INFO] svc-{i%9} node-{i%5}: routine heartbeat ok "
            f"seq={i} latency={40+(i*7)%300}ms region=us-{i%3} status=200")

def build(nlines, depth):
    lines = [filler_line(i) for i in range(nlines)]
    lines.insert(int(len(lines) * depth), NEEDLE)
    return ("You are reviewing a long server log. Somewhere in it is one special fact.\n\n"
            + "\n".join(lines) +
            "\n\nQUESTION: What is the access code for building 7? Answer with ONLY the code.")

def ntok(text):
    # (count, max_model_len) via /tokenize; 실패 시 (None, None)로 폴백 신호
    try:
        d = post(TOK, {"model": M, "prompt": text}, timeout=120)
        return d.get("count"), d.get("max_model_len")
    except Exception:
        return None, None

label = sys.argv[1] if len(sys.argv) > 1 else "niah"
TGT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
depths = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.1", "0.5", "0.9"])]

# --- 라인당 토큰수 캘리브레이션 (정확) 또는 char-추정 폴백 ---
SAMPLE = 2000
sc, mml = ntok(build(SAMPLE, 0.5))
cap = mml or 262144
if sc:
    tpl = sc / SAMPLE; mode = "tokenizer"
else:
    tpl = (len(build(SAMPLE, 0.5)) / SAMPLE) / 2.8; mode = "char-est"  # 폴백
target = (TGT if TGT > 0 else cap)
target = min(target, cap - 256)            # 질문+출력 여유
nlines = max(100, int(target / tpl))
# /tokenize 가능하면 캡 아래로 정밀 조정
if sc:
    pc, _ = ntok(build(nlines, 0.5))
    while pc and pc > cap - 64 and nlines > 100:
        nlines = int(nlines * 0.95); pc, _ = ntok(build(nlines, 0.5))
    achieved = pc
else:
    achieved = int(nlines * tpl)
print(f"[{label}] mode={mode} tpl={tpl:.2f} cap={cap} nlines={nlines} ~{achieved} tokens", flush=True)

ok = 0; maxtok = 0
for d in depths:
    try:
        res = post(URL, {"model": M, "messages": [{"role": "user", "content": build(nlines, d)}],
                         "max_tokens": 512, "temperature": 0})
        a = res["choices"][0]["message"]["content"]; pin = res.get("usage", {}).get("prompt_tokens", 0)
    except Exception as e:
        a = f"ERR:{e}"[:60]; pin = 0
    hit = CODE in a; ok += hit; maxtok = max(maxtok, pin)
    print(f"[{label}] depth={d:.0%} prompt_tokens={pin} hit={hit} ans={a.strip()[:40]}", flush=True)
print(f"[{label}] NIAH {ok}/{len(depths)} (max prompt_tokens={maxtok}, cap={cap})", flush=True)
