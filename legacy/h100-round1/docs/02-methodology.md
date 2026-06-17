# 측정 방법론

## 1. 왜 non-streaming 총처리량인가 (핵심)

> **모든 처리량은 `출력토큰수 / 전체 wall-clock`으로만 측정한다. streaming 토큰율은 금지.**

| 지표 | 정의 | 성질 |
|---|---|---|
| non-streaming 총처리량 (채택) | 출력토큰 / 전체 latency | 토큰의 시간적 분포에 **불변** |
| streaming `decode_tok/s` (금지) | 디코드 구간 토큰 흐름 속도 | 버스트에 **취약** → spec decode 왜곡 |

**이유 — speculative decoding은 버스트로 토큰을 뱉는다.**
MTP/spec decode는 "각 forward pass를 빠르게" 만드는 게 아니라 "한 forward pass가 토큰을 여러 개
뱉게" 만든다. 한 스텝: 드래프터가 γ개 추측(쌈) → 타깃 모델이 γ+1 위치를 forward pass 1회로 검증(비쌈)
→ 채택분 k+1개를 **한꺼번에 커밋**. 따라서 토큰은 `버스트(k+1개 동시) → 갭(검증 1회) → 버스트`로 나오고,
순간 토큰율은 무한대~0을 오간다. 여기에 단일 "decode_tok/s"를 붙이면 평균 방식에 따라 값이 크게 흔들린다.

```
AR:          |--T--|t|--T--|t|--T--|t|        균일 (1 pass = 1 token)
spec decode: |--T--|ttt|------T------|tttt|    버스트 (1 pass = 여러 token)
```

이 결함이 초기 "MTP 손해" 결론을 만들었고(특히 Qwen/A100 측정에서 ~3배 과소측정 → "0.44x 손해"),
non-streaming 재측정으로 **2.6~3.0x 이득**으로 뒤집혔다.

## 2. acceptance length 항등식 — 측정을 검증한 불변량

vLLM이 보고하는 **Mean acceptance length A** = 타깃 forward pass 1회당 평균 확정 토큰 수.

- AR: 1토큰/pass → 처리량 1/T
- spec: A토큰/pass(시간 ≈ T) → 처리량 **A/T** → **가속 ≈ A**

즉 acceptance length가 곧 (근사) 가속 배수다. streaming이 "손해"라는데 A≈3이면 **논리적 모순** —
A는 엔진 내부 카운트(타이밍 무관, 위조 불가)이므로 거짓말하는 쪽은 streaming 지표로 확정된다.
이 도구-독립적 교차검증이 측정 결함을 잡아낸 결정적 단서였다.
(단서: 드래프터 오버헤드로 실제 가속 ≤ A, 고동시성에선 검증 연산 증가로 손해 가능 —
vLLM `--speculative-disable-by-batch-size`가 이를 공식 인정. 단일스트림·저동시성에선 가속 ≈ A.)

## 3. 측정 통제 (공정성)

`bench_tp.py` (회수본: `../benchmarks/bench_tp.py`):
- **출력 512토큰 고정** (`ignore_eos:True`, `max_tokens:512`) — 길이 변동·조기 EOS로 인한 비교 오염 제거
- **`temperature:0`** — 결정적
- **prefill 분리** — short + 8K(~8000토큰) 두 컨텍스트 각각 측정
- **warmup 1회** — cudagraph 컴파일/콜드스타트를 측정 밖으로(특히 diffusion 필수)
- 서버 **`--no-enable-prefix-caching`** — 캐시 히트로 인한 거짓 가속 제거

## 4. 정확도 태스크 정의

`bench_acc.py` (회수본: `../benchmarks/bench_acc.py`), N=30:
- **합성 인시던트 로그** 생성 — 120줄 중 한 서비스가 먼저 실패(OOM/disk full/segfault/deadlock/cert expired),
  의존 서비스들이 이후 `upstream timeout calling {root}`로 cascade, 나머지는 정상 노이즈(200 OK).
- 모델에게 "근본원인 서비스를 보기(셔플된 root+deps) 중 **이름만** 답하라" 요청.
- **채점**: ground-truth 서비스명 substring 매칭.
- 실 ops 분석 태스크(SRE 인시던트 근본원인)와 동형이 되도록 설계.
- **한계**: 채점 호출이 `max_tokens=24`라, 답 앞에 추론 서두를 붙이는 모델은 이름이 잘림 → Qwen
  아티팩트의 원인(`04-troubleshooting.md` #10). 멀티모델 비교 시 토큰예산·추출기를 모델포맷에 강건하게 해야 함.

## 5. 환경 / 이미지

| 용도 | 이미지 | 비고 |
|---|---|---|
| Gemma4 AR/MTP, Qwen | `vllm/vllm-openai:nightly` | gemma4_unified 네이티브. stable 0.22.1은 폴백 후 크래시 |
| diffusion | `vllm/vllm-openai:gemma` | block diffusion(256토큰 canvas) |

MTP 설정: Gemma=별도 드래프터(`gemma-4-<size>-it-assistant`), Qwen=임베디드 MTP 헤드.
재현 상세는 `05-reproduce.md`.
