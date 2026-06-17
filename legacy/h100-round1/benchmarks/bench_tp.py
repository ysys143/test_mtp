#!/usr/bin/env python3
# 공정 지표: total 처리량 = 출력토큰/전체요청시간 (AR·diffusion 공통). prefix cache off 전제.
import urllib.request, json, time, sys
BASE='http://localhost:8000'; URL=BASE+'/v1/chat/completions'
def detect():
    with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as r: return json.load(r)['data'][0]['id']
M=detect()
def req(content,mt):
    p=json.dumps({'model':M,'messages':[{'role':'user','content':content}],'max_tokens':mt,'temperature':0,'ignore_eos':True}).encode()
    r=urllib.request.Request(URL,data=p,headers={'Content-Type':'application/json'}); t0=time.perf_counter()
    with urllib.request.urlopen(r,timeout=900) as resp:
        d=json.load(resp); u=d.get('usage',{})
    return (time.perf_counter()-t0)*1000, u.get('completion_tokens',0), u.get('prompt_tokens',0)
def bigtext(n):
    line='2026 [INFO] svc-%d pod-1 node-2: req handled status=200 lat=%dms trace=ab%d; '
    s='Analyze these logs and summarize:\n'
    i=0
    while len(s)<int(n*1.83): s+=line%(i%9,(i*17)%1900,i); i+=1
    return s
label=sys.argv[1] if len(sys.argv)>1 else 'run'
print(f'[{label}] model={M}',flush=True)
for cn,prompt in [('short','Write a long detailed technical document about distributed systems, databases, and consensus.'),
                  ('8K',bigtext(8000)+'\nGive a structured analysis and 3 recommendations.')]:
    req(prompt,16)  # warmup (cudagraph)
    lat,out,pin=req(prompt,512)
    tp=out/(lat/1000) if lat>0 else 0
    print(f'  [{cn}] pin={pin} total_tp={tp:.1f} tok/s (out={out}, lat={lat:.0f}ms)',flush=True)
