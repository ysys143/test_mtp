#!/usr/bin/env python3
# ops 정확도 v2: max_tokens 24->256 + "마지막 언급 서비스명" 추출(추론 서두 대응)
# 프롬프트는 bench_acc.py와 동일 -> Gemma 거동 불변(하위호환)
import urllib.request, json, sys
BASE="http://localhost:8000"; URL=BASE+"/v1/chat/completions"
def detect():
    with urllib.request.urlopen(BASE+"/v1/models",timeout=10) as r: return json.load(r)["data"][0]["id"]
M=detect()
SVCS=["fastapi-auth","fastapi-chat","fastapi-rag","celery","mongodb","redis","qdrant","keycloak","nginx","postgres","kafka","elasticsearch"]
ROOT_ERR=["OOMKilled: container exceeded memory limit","disk full: no space left on device","segfault: process crashed","deadlock detected, transaction aborted","certificate expired, TLS handshake failed"]
def incident(seed):
    root=SVCS[(seed*7)%len(SVCS)]
    deps=[SVCS[(seed*7+1+j)%len(SVCS)] for j in range(3)]
    deps=[d for d in deps if d!=root][:3]
    err=ROOT_ERR[seed%len(ROOT_ERR)]
    lines=[]; b=1781160000+seed*3600
    for i in range(120):
        t=b+i*5
        if i==40: lines.append(f"{t} [ERROR] {root} pod-1: {err}")
        elif i>40 and i%7==0 and deps:
            d=deps[(i//7)%len(deps)]
            lines.append(f"{t} [ERROR] {d} pod-2: upstream timeout calling {root} (5xx), retry failed")
        else:
            sv=SVCS[(i*3+seed)%len(SVCS)]
            lines.append(f"{t} [INFO] {sv} pod-{(i%4)+1}: request handled 200 lat={50+(i*13%400)}ms")
    choices=[root]+deps
    order=sorted(range(len(choices)), key=lambda k:(k*13+seed)%97)
    shuf=[choices[k] for k in order]
    joined=", ".join(shuf)
    prompt=("Below are Kubernetes logs during an incident.\n\n=== LOGS ===\n"+"\n".join(lines)+
            "\n\n=== QUESTION ===\nWhich service is the ROOT CAUSE (the service that failed first and caused others to fail)? "
            "Answer with ONLY the exact service name, one of: "+joined+". Output just the name, nothing else.")
    return prompt, root
def ask(prompt):
    p=json.dumps({"model":M,"messages":[{"role":"user","content":prompt}],"max_tokens":256,"temperature":0}).encode()
    r=urllib.request.Request(URL,data=p,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip().lower()
def extract(ans):
    best=None; bestpos=-1
    for s in SVCS:
        pos=ans.rfind(s.lower())
        if pos>bestpos: bestpos=pos; best=s
    return best
label=sys.argv[1] if len(sys.argv)>1 else "run"
N=int(sys.argv[2]) if len(sys.argv)>2 else 30
ok=0; bad=[]
for i in range(N):
    pr,gt=incident(i)
    try: a=ask(pr)
    except Exception as e: a=f"ERR:{e}"[:40]
    pred=extract(a)
    if pred==gt: ok+=1
    else: bad.append((gt,pred,a[:50]))
print(f"[{label}] model={M} accuracy={ok}/{N} = {ok/N*100:.1f}% (bench_acc2)",flush=True)
if bad[:3]: print("  오답 예:", bad[:3],flush=True)
