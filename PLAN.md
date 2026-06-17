# PLAN — 컨텍스트 길이 안정화 → 다면 평가 배터리

**작성:** 2026-06-14 · **상태:** Phase 0 진행 중

## 0. 배경 / 동기

- **"8K"는 H100 한계가 아니라 측정 선택이었다.** 현 `bench_tp.py`는 short + `bigtext(8000)` 두 점만 쟀고,
  서버 `max-model-len`도 16384(31B는 12288)로 **보수적으로** 잡았다. 즉 8K는 우리가 고른 테스트 프롬프트
  크기일 뿐, 하드웨어가 그 이상 못 한다는 뜻이 아니다.
- Gemma4 / Qwen3.6은 네이티브 **128K+** 컨텍스트를 지원한다. H100 80GB가 이 규모 모델을 8K밖에 못
  뽑는다는 건 사실이 아니며, **실제 감당 가능한 최대 컨텍스트를 제대로 규명·고정**해야 한다.
- 장문맥 벤치(RULER 등)는 컨텍스트가 안정돼야 의미가 있으므로 **평가 배터리보다 선행**한다.

## 1. 단계 개요 (순서 고정)

| Phase | 내용 | 게이트 |
|---|---|---|
| **0** | 현재 측정 마무리 (Qwen27 stable + Gemma 컨트롤) | 결과 수령 시 Phase 1 |
| **1** | 현 하네스 + 결과 전부 `legacy/`로 이관 (클린 슬레이트) | 이관 완료 시 Phase 2 |
| **2** | **컨텍스트 길이 안정화** — (모델×정밀도)별 최대 안정 컨텍스트 규명·고정 | **완전 안정 후** Phase 3 |
| **3** | 다면 평가 배터리 — lm-eval / inspect_ai / HRET | Phase 2 고정값 사용 |

> Phase 2가 **완전히 안정되기 전엔 Phase 3로 넘어가지 않는다.**

---

## Phase 0 — 현재 측정 마무리 (진행 중)

- Qwen27-FP8 base/MTP를 stable `vllm/vllm-openai:v0.22.1` 이미지로 측정(nightly 비호환 우회).
- Gemma 26B-fp8을 신규 채점기 `bench_acc2`로 재채점 → 100% 유지 확인(채점기 공정성 검증).
- Qwen35 정확도는 재측정 완료: **83.3%**(신규 채점기) vs 6.7%(옛 채점기 아티팩트).
- 산출 후 `01-results.md`/CSV에 반영하고 Phase 1로.

## Phase 1 — 레거시 이관 (클린 슬레이트)

- 현 테스트 하네스(`benchmarks/bench_tp.py`·`bench_acc*.py`·`run_*.sh`)와 결과(`docs/01~05`,
  `data/results.csv`)를 **전부 `legacy/`로** 이관. 기존 `legacy/`는 그대로 둠(중첩 OK).
- 이유: 다음 단계는 (a) 더 긴 컨텍스트, (b) 표준 배터리 기반이라 측정 패러다임이 바뀐다. 과거 결과는
  감사 추적으로만 보존하고 새 결과와 섞지 않는다.
- 남기는 것: 본 `PLAN.md`, 새 작업 디렉토리 골격(`context/`, `evals/`).

## Phase 2 — 컨텍스트 길이 안정화 (핵심)

**목표:** 각 (모델 × 정밀도)에서 H100 80GB에 **안정 서빙되는 최대 컨텍스트**를 규명하고 고정한다.
"로드된다"가 아니라 "그 길이에서 실제로 정확히 동작한다"까지 검증한다.

### 2.1 메모리 모델 (왜 길이가 갈리나)
```
VRAM(80GB×util) = 가중치 + KV캐시 + 활성/오버헤드 + cudagraph
KV per token   = 2(K,V) × layers × kv_heads × head_dim × dtype_bytes   (GQA면 kv_heads 작아 KV 작음)
최대 토큰수     = (VRAM − 가중치 − 오버헤드) / KV_per_token
```
→ **정밀도를 낮추면(fp8/int8/qat) 가중치가 작아져 KV 여유 = 컨텍스트가 늘어난다.** 즉 기존 정밀도 축과
컨텍스트 축이 직접 연결된다.

### 2.2 레버 (컨텍스트 최대화)
- **저정밀 가중치**(fp8/int8/qat) → KV 여유 확보 (가장 큰 레버)
- **`--kv-cache-dtype fp8`** → KV 절반 → 컨텍스트 ~2배
- `--gpu-memory-utilization` 상향(0.95) + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
- `--enable-chunked-prefill` (장문 prefill 메모리/throughput)
- 네이티브 max 초과가 필요하면 `--rope-scaling`(정확도 영향 검증 필요)

### 2.3 절차
1. 모델 config에서 **네이티브 최대 컨텍스트** 확인.
2. `max-model-len`을 단계적으로 올리며 **OOM 직전 최대치** 탐색(이분 탐색).
3. 그 길이에서 **NIAH(needle-in-a-haystack) 검증** — 로드 ≠ 작동. 컨텍스트 곳곳에 심은 정보를 실제로
   회수하는지 확인.
4. (모델×정밀도)별 **최대 안정 컨텍스트**를 `context/max_context.csv`에 기록.
5. 공정 비교용 **공통 컨텍스트 길이** 결정(최소공통 또는 티어: 예 16K/32K/64K/128K).

### 2.4 산출물
- `context/probe_maxlen.sh` — 셀별 max-model-len 이분 탐색 러너
- `context/niah.py` — 길이별 needle 회수 검증
- `context/max_context.csv` — (model, precision, max_stable_ctx, kv_dtype, util, niah_pass)
- **완료 기준:** 각 셀이 목표 길이로 크래시 없이 서빙 + NIAH 통과. 이게 충족돼야 Phase 3.

## Phase 3 — 다면 평가 배터리 (안정화 후에만)

Phase 2에서 고정한 컨텍스트로, 같은 vLLM OpenAI 엔드포인트에 세 도구를 붙인다.

| 축 | 도구 | 대표 벤치 |
|---|---|---|
| 영어 추론·장문맥 | **lm-eval** | GPQA-Diamond, MMLU-Pro, BBH, RULER(고정 길이) |
| Tool / 에이전트 / 자작 ops | **inspect_ai** | BFCL/τ-bench, 난도 상향 ops 태스크 |
| 한국어 | **HRET** | KMMLU, HAE-RAE Bench, HRM8K |

- 비교 잣대 통일: temp 0, few-shot/프롬프트 고정, 동일 컨텍스트.
- 정밀도×기법 축 유지(MTP는 무손실이라 정확도는 정밀도/길이 비교용).
- 한국어·영어 점수는 **직접 비교 금지**(각각의 보존 여부로 해석).

---

## 새 디렉토리 구조 (안)

```
test_mtp/
├── PLAN.md                  # 본 문서
├── context/                 # Phase 2
│   ├── probe_maxlen.sh
│   ├── niah.py
│   └── max_context.csv      # 고정된 최대 안정 컨텍스트
├── evals/                   # Phase 3
│   ├── lm_eval/             # 러너 + 설정
│   ├── inspect/             # 태스크 + 러너
│   └── hret/                # 한국어
├── README.md                # Phase 3 후 갱신
└── legacy/                  # 현 하네스·결과 포함 과거 전부
```

## 미해결/리스크
- v0.22.1(Qwen27)과 nightly(Gemma4) 이미지가 다름 → 컨텍스트 안정화도 **이미지별로** 따로 해야 함.
- RoPE scaling으로 네이티브 초과 시 장문 정확도 저하 가능 → NIAH로 반드시 검증.
- 단일 H100 80GB라 31B-bf16처럼 가중치 큰 셀은 장문에서 더 빨리 막힘 → 저정밀이 컨텍스트에도 유리.
