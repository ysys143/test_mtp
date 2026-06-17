#!/usr/bin/env python3
# ops 정확도: 로그 cascade에서 근본원인 서비스 식별 (ground truth 채점)
import urllib.request, json, sys
BASE='http://localhost:8000'; URL=BASE+'/v1/chat/completions'
def detect():
    with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as r: return json.load(r)['data'][0]['id']
M=detect()
SVCS=['fastapi-auth','fastapi-chat','fastapi-rag','celery','mongodb','redis','qdrant','keycloak','nginx','postgres','kafka','elasticsearch']
ROOT_ERR=['OOMKilled: container exceeded memory limit','disk full: no space left on device','segfault: process crashed','deadlock detected, transaction aborted','certificate expired, TLS handshake failed']
def incident(seed):
    root=SVCS[(seed*7)%len(SVCS)]
    deps=[SVCS[(seed*7+1+j)%len(SVCS)] for j in range(3)]
    deps=[d for d in deps if d!=root][:3]
    err=ROOT_ERR[seed%len(ROOT_ERR)]
    lines=[]; b=1781160000+seed*3600
    # 노이즈 + root 원인 + dependents의 cascade(=root로의 timeout)
    for i in range(120):
        t=b+i*5
        if i==40:
            lines.append(f'{t} [ERROR] {root} pod-1: {err}')
        elif i>40 and i%7==0 and deps:
            d=deps[(i//7)%len(deps)]
            lines.append(f'{t} [ERROR] {d} pod-2: upstream timeout calling {root} (5xx), retry failed')
        else:
            sv=SVCS[(i*3+seed)%len(SVCS)]
            lines.append(f'{t} [INFO] {sv} pod-{(i%4)+1}: request handled 200 lat={50+(i*13%400)}ms')
    choices=[root]+deps
    # 선택지 셔플(결정적)
    order=sorted(range(len(choices)), key=lambda k:(k*13+seed)%97)
    shuf=[choices[k] for k in order]
    prompt=('Below are Kubernetes logs during an incident.\n\n=== LOGS ===\n'+'\n'.join(lines)+
            f'\n\n=== QUESTION ===\nWhich service is the ROOT CAUSE (the service that failed first and caused others to fail)? '
            f'Answer with ONLY the exact service name, one of: {", ".join(shuf)}. Output just the name, nothing else.')
    return prompt, root
def ask(prompt):
    p=json.dumps({'model':M,'messages':[{'role':'user','content':prompt}],'max_tokens':24,'temperature':0}).encode()
    r=urllib.request.Request(URL,data=p,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=300) as resp:
        return json.load(resp)['choices'][0]['message']['content'].strip().lower()
label=sys.argv[1] if len(sys.argv)>1 else 'run'
N=int(sys.argv[2]) if len(sys.argv)>2 else 30
ok=0; bad=[]
for i in range(N):
    pr,gt=incident(i)
    try: a=ask(pr)
    except Exception as e: a=f'ERR:{e}'[:30]
    if gt.split('-')[-1] in a or gt in a: ok+=1
    else: bad.append((gt,a[:30]))
print(f'[{label}] model={M} accuracy={ok}/{N} = {ok/N*100:.1f}%',flush=True)
if bad[:3]: print('  오답 예:', bad[:3],flush=True)
