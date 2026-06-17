#!/usr/bin/env python3
# non-streaming 순수 decode 측정 (prefill 분리, ignore_eos로 512 강제) + acceptance
import urllib.request, json, time, sys
BASE='http://localhost:8000'; URL=BASE+'/v1/chat/completions'; MET=BASE+'/metrics'
def detect():
    with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as r: return json.load(r)['data'][0]['id']
M=detect()
def counters():
    try:
        with urllib.request.urlopen(MET,timeout=10) as r: t=r.read().decode()
    except Exception: return 0,0
    a=dr=0.0
    for ln in t.splitlines():
        if ln.startswith('vllm:spec_decode_num_accepted_tokens_total'): a=float(ln.split()[-1])
        elif ln.startswith('vllm:spec_decode_num_draft_tokens_total'): dr=float(ln.split()[-1])
    return a,dr
def block(content, mt, ieos=True):
    p=json.dumps({'model':M,'messages':[{'role':'user','content':content}],'max_tokens':mt,'temperature':0,'ignore_eos':ieos}).encode()
    r=urllib.request.Request(URL,data=p,headers={'Content-Type':'application/json'}); t0=time.perf_counter()
    with urllib.request.urlopen(r,timeout=600) as resp:
        d=json.load(resp); u=d.get('usage',{})
    return (time.perf_counter()-t0)*1000, u.get('completion_tokens',0), u.get('prompt_tokens',0)
label=sys.argv[1] if len(sys.argv)>1 else 'run'
SHORT='Write a long detailed technical document about distributed systems, consensus algorithms, and database internals.'
block(SHORT,64)  # warmup
lp,_,pin=block(SHORT,1)               # prefill 분리
a0,d0=counters()
lf,out,_=block(SHORT,512)             # 512 강제
a1,d1=counters()
dec=(out-1)/((lf-lp)/1000) if lf>lp and out>1 else 0
dd=d1-d0
acc=(a1-a0)/dd*100 if dd>0 else None
if acc is not None:
    print(f'[{label}] model={M} decode_tok/s={dec:.1f} acceptance={acc:.1f}% (out={out}, prefill={lp:.0f}ms)',flush=True)
else:
    print(f'[{label}] model={M} decode_tok/s={dec:.1f} (out={out}, prefill={lp:.0f}ms) NON-MTP',flush=True)
