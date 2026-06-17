#!/usr/bin/env python3
"""Gemma 4 변종 비교 v3 — 2컨텍스트(8K/128K) x 변종별 max 동시성.
single은 streaming(TTFT), 동시성은 non-streaming(서버 throughput 정확)."""
import urllib.request, json, time, sys, statistics
from concurrent.futures import ThreadPoolExecutor

BASE='http://localhost:8000'; URL=BASE+'/v1/chat/completions'; METRICS=BASE+'/metrics'
MAXTOK=1024; CPT=1.83  # chars per token (측정값)
CTXS=[8000, 123000]    # 8K, 128K-class

def detect_model():
    with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as r:
        return json.load(r)['data'][0]['id']
MODEL=detect_model()

def build_prompt(seed, ctx_tokens):
    char_target=int(ctx_tokens*CPT)
    services=['fastapi-auth','fastapi-chat','fastapi-rag','celery','mongodb','redis',
              'qdrant','keycloak','nginx','envoy','postgres','kafka','elasticsearch']
    levels=['INFO','INFO','INFO','DEBUG','WARN','ERROR']
    msgs={'INFO':['request handled 200','cache hit','health ok','connected','enqueued','token refreshed','indexed doc','committed offset'],
          'DEBUG':['span started','ctx propagated','pool checkout','gc minor'],
          'WARN':['slow query 1.2s','pool 85% util','retry 2/3','mem pressure','rate limit near','disk 78%'],
          'ERROR':['pool exhausted','upstream timeout 30s','OOMKilled','5xx upstream','breaker open','lock acquire fail','replica lag 12s']}
    lines=[]; b=1781160000+seed*7200; clen=0; i=0
    while clen<char_target:
        lv=levels[(i*7+seed)%len(levels)]; sv=services[(i*3+seed)%len(services)]
        m=msgs[lv][(i+seed)%len(msgs[lv])]
        ln=(f'{b+i*3} [{lv}] {sv} pod-{(i%6)+1} node-{(i%4)+1}: {m} '
            f'(trace={i:06d}{seed:02d}, span={(i*13)%99999}, lat={50+(i*17%1900)}ms, '
            f'req=/api/v{(i%3)+1}/{sv}/op{i%50})')
        lines.append(ln); clen+=len(ln)+1; i+=1
    return ('You are an SRE incident analysis assistant. Below is an observability dump.\n\n'
            '=== LOGS ===\n'+'\n'.join(lines)+'\n\n=== TASK ===\n'
            'Identify the top 3 incidents (root cause, blast radius, severity SEV1-4) and a remediation plan.')

def run_stream(prompt):
    payload=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],
        'max_tokens':MAXTOK,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}).encode()
    req=urllib.request.Request(URL,data=payload,headers={'Content-Type':'application/json'})
    t0=time.perf_counter(); t_first=None; tokens=0; ptoks=None
    try:
        with urllib.request.urlopen(req,timeout=1200) as resp:
            for line in resp:
                line=line.decode().strip()
                if not line.startswith('data:'): continue
                d=line[5:].strip()
                if d=='[DONE]': break
                o=json.loads(d)
                if o.get('usage'): ptoks=o['usage'].get('prompt_tokens',ptoks)
                ch=o.get('choices') or []
                if ch and ch[0].get('delta',{}).get('content'):
                    if t_first is None: t_first=time.perf_counter()
                    tokens+=1
    except Exception as e:
        return {'ttft':-1,'ttlp':-1,'tps':0,'ptoks':ptoks,'err':str(e)[:80]}
    t_end=time.perf_counter()
    return {'ttft':(t_first-t0)*1000 if t_first else -1,'ttlp':(t_end-t0)*1000,
            'tps':tokens/(t_end-t_first) if t_first and tokens else 0,'ptoks':ptoks,'tokens':tokens}

def run_block(prompt):
    payload=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],
        'max_tokens':MAXTOK,'temperature':0}).encode()
    req=urllib.request.Request(URL,data=payload,headers={'Content-Type':'application/json'})
    t0=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=1800) as resp:
            d=json.load(resp); u=d.get('usage',{})
            return {'lat':(time.perf_counter()-t0)*1000,'out':u.get('completion_tokens',0),'pin':u.get('prompt_tokens',0)}
    except Exception as e:
        return {'lat':-1,'out':0,'pin':0,'err':str(e)[:80]}

def spec_metrics():
    try:
        with urllib.request.urlopen(METRICS,timeout=10) as r: text=r.read().decode()
        acc=draft=0.0
        for ln in text.splitlines():
            if ln.startswith('vllm:spec_decode_num_accepted_tokens_total'): acc=float(ln.split()[-1])
            elif ln.startswith('vllm:spec_decode_num_draft_tokens_total'): draft=float(ln.split()[-1])
        return (acc,draft) if draft>0 else None
    except Exception: return None

label=sys.argv[1] if len(sys.argv)>1 else 'run'
KV_TOTAL=int(sys.argv[2]) if len(sys.argv)>2 else 800000
print(f'[{label}] model={MODEL} KV_total={KV_TOTAL} temp=0 max_tok={MAXTOK}',flush=True)

for ctx in CTXS:
    conc=max(2, min(128, round(KV_TOTAL/ctx*1.1)))   # 변종별 max에 맞춤
    s=run_stream(build_prompt(0,ctx))
    print(f'  [ctx={ctx//1000}K single] ptok={s.get("ptoks")} TTFT={s["ttft"]:.0f}ms decode_tok/s={s["tps"]:.1f} out={s.get("tokens")}',flush=True)
    prompts=[build_prompt(100+i,ctx) for i in range(conc)]
    w0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex: rs=list(ex.map(run_block,prompts))
    w=time.perf_counter()-w0
    ok=[x for x in rs if x['out']>0]; tot=sum(x['out'] for x in ok)
    lats=[x['lat'] for x in ok]
    if ok:
        print(f'  [ctx={ctx//1000}K conc={conc}] done={len(ok)}/{conc} agg_out_tok/s={tot/w:.1f} '
              f'wall={w:.1f}s lat_avg={statistics.mean(lats):.0f}ms lat_p99={max(lats):.0f}ms',flush=True)
    else:
        print(f'  [ctx={ctx//1000}K conc={conc}] ALL FAILED ({rs[0].get("err","?") if rs else "?"})',flush=True)

sm=spec_metrics()
print(f'  [spec] acceptance={sm[0]/sm[1]*100:.1f}% (acc={sm[0]:.0f}/draft={sm[1]:.0f})' if sm else '  [spec] non-MTP',flush=True)
