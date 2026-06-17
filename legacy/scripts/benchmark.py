#!/usr/bin/env python3
"""모델 자동감지 + temp=0 + 동시성 스윕 MTP 비교. (27B dense / A3B 공용)"""
import urllib.request, json, time, sys, statistics
from concurrent.futures import ThreadPoolExecutor

BASE    = 'http://localhost:8000'
URL     = BASE + '/v1/chat/completions'
METRICS = BASE + '/metrics'
MAXTOK  = 512

def detect_model():
    with urllib.request.urlopen(BASE + '/v1/models', timeout=10) as r:
        return json.load(r)['data'][0]['id']

MODEL = detect_model()

def build_incident_prompt(seed):
    services = ['fastapi-auth','fastapi-chat','fastapi-rag','celery-worker',
                'mongodb','redis','qdrant','keycloak','nginx-ingress']
    levels = ['INFO','INFO','INFO','WARN','ERROR']
    msgs = {
        'INFO': ['request handled','cache hit','health check ok','connection established',
                 'task enqueued','token refreshed','document indexed'],
        'WARN': ['slow query 1.2s','connection pool 85% utilized','retry attempt 2/3',
                 'memory pressure detected','rate limit approaching'],
        'ERROR':['connection pool exhausted','upstream timeout after 30s',
                 'OOMKilled: container exceeded memory limit','5xx from upstream',
                 'circuit breaker open','failed to acquire lock'],
    }
    lines=[]; base=1781160000+seed*3600
    for i in range(160):
        lvl=levels[(i*7+seed)%len(levels)]; svc=services[(i*3+seed)%len(services)]
        m=msgs[lvl][(i+seed)%len(msgs[lvl])]; ts=base+i*11
        lines.append(f'{ts} [{lvl}] {svc} pod-{(i%4)+1}: {m} '
                     f'(trace_id=abc{i:03d}{seed}, latency_ms={50+(i*13%800)})')
    metrics='\n'.join([f'{svc}: cpu_p95={30+(j*7%60)}% mem_p95={40+(j*11%55)}% '
                       f'rps={100+j*37} error_rate={(j*3%12)}.{j%10}% p99_ms={120+j*45}'
                       for j,svc in enumerate(services)])
    return ('You are an SRE incident analysis assistant. Below is observability data '
            'from a Kubernetes cluster during a suspected incident window.\n\n'
            '=== APPLICATION LOGS ===\n'+'\n'.join(lines)+'\n\n'
            '=== METRIC BASELINE DEVIATIONS ===\n'+metrics+'\n\n'
            '=== TASK ===\n1. Identify root cause and failure propagation chain.\n'
            '2. List affected services in blast-radius order.\n'
            '3. Recommend 3 remediation steps.\n4. Estimate severity (SEV1-SEV4).\n'
            'Provide a structured incident report.')

def run_once(prompt):
    payload=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],
        'max_tokens':MAXTOK,'temperature':0,'stream':True,
        'chat_template_kwargs':{'enable_thinking':False}}).encode()
    req=urllib.request.Request(URL,data=payload,headers={'Content-Type':'application/json'})
    t0=time.perf_counter(); t_first=None; tokens=0
    with urllib.request.urlopen(req,timeout=300) as resp:
        for line in resp:
            line=line.decode().strip()
            if not line.startswith('data:'): continue
            d=line[5:].strip()
            if d=='[DONE]': break
            try:
                delta=json.loads(d)['choices'][0]['delta']
                if delta.get('content'):
                    if t_first is None: t_first=time.perf_counter()
                    tokens+=1
            except Exception: pass
    t_end=time.perf_counter()
    return {'ttft':(t_first-t0)*1000 if t_first else -1,
            'ttlp':(t_end-t0)*1000,'tokens':tokens,
            'tps':tokens/(t_end-t_first) if t_first and tokens else 0}

def spec_metrics():
    try:
        with urllib.request.urlopen(METRICS,timeout=10) as r: text=r.read().decode()
        acc=draft=0.0
        for ln in text.splitlines():
            if ln.startswith('vllm:spec_decode_num_accepted_tokens_total'): acc=float(ln.split()[-1])
            elif ln.startswith('vllm:spec_decode_num_draft_tokens_total'): draft=float(ln.split()[-1])
        return acc,draft
    except Exception: return None

label=sys.argv[1] if len(sys.argv)>1 else 'run'
approx_in=len(build_incident_prompt(0))//4
print(f'[{label}] model={MODEL} temp=0 max_tok={MAXTOK} ~{approx_in} input tokens',flush=True)
_=run_once(build_incident_prompt(0))

ss=[run_once(build_incident_prompt(s)) for s in range(3)]
print(f'  [single] TTFT avg={statistics.mean(r["ttft"] for r in ss):.0f}ms  '
      f'TTLP avg={statistics.mean(r["ttlp"] for r in ss):.0f}ms  '
      f'tok/s avg={statistics.mean(r["tps"] for r in ss):.1f}  out={ss[0]["tokens"]}',flush=True)

for C in (8,16):
    prompts=[build_incident_prompt(100+C*10+i) for i in range(C)]
    wall0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=C) as ex:
        rs=list(ex.map(run_once,prompts))
    wall=time.perf_counter()-wall0
    tot=sum(r['tokens'] for r in rs)
    print(f'  [conc={C}] agg_tok/s={tot/wall:.1f}  '
          f'TTFT avg={statistics.mean(r["ttft"] for r in rs):.0f}ms  '
          f'TTLP avg={statistics.mean(r["ttlp"] for r in rs):.0f}ms  '
          f'wall={wall:.1f}s total_out={tot}',flush=True)

sm=spec_metrics()
if sm and sm[1]>0:
    acc,draft=sm
    print(f'  [spec] accepted={acc:.0f} draft={draft:.0f} acceptance={acc/draft*100:.1f}%',flush=True)
else:
    print('  [spec] no spec-decode metrics (MTP off)',flush=True)
