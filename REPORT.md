# 보고서 — Phase 1·2 (재구조화 + 컨텍스트 길이 안정화)

> **[통합] 본 문서는 전체 여정 종합본 `report/`로 대체됨 → [report/README.md](report/README.md) (Phase1·2는 report/01·03).** 아래는 원본 보존.

**대상:** Gemma 4 (12B·26B-A4B·31B) / Qwen3.6 (27B·35B-A3B) · **환경:** 단일 H100 80GB (GCP), vLLM
**범위:** Phase 1(레포 재구조화) + Phase 2(컨텍스트 한계 규명·검증). Phase 3(다면 평가)은 방법론 재검토 중(§상태).
**작성:** 2026-06-14

---

## Phase 1 — 재구조화 (클린 슬레이트)

### 동기
1차 H100 라운드 산출물(속도·정확도 보고서)이 측정 결함 발견마다 정정(취소선·"무효"·vN)을 덧대 비대해졌다.
다음 단계(더 긴 컨텍스트, 표준 평가 배터리)는 측정 패러다임이 달라, 과거 결과를 감사 추적으로만 보존하고
새 작업과 분리하기로 했다.

### 한 일
- 1차 H100 라운드 전체(`benchmarks/`·`docs/`·`README`·`data/`)를 `legacy/h100-round1/`로 이관.
- 초기 A100 라운드 보고서·로그는 `legacy/`에 그대로 보존.
- 새 작업 공간 `context/`(Phase 2)·`evals/`(Phase 3) 신설. 진입점 `README.md`·계획 `PLAN.md` 재작성.

### 현재 구조
```
test_mtp/
├── PLAN.md, README.md, REPORT.md(본 문서)
├── context/   — Phase 2 (probe_maxlen.sh, niah.py, max_context.csv, FINDINGS.md)
├── evals/     — Phase 3 (lm-eval 러너 등, 진행 중)
└── legacy/    — h100-round1/ (1차 H100) + 초기 A100 산출물
```

---

## Phase 2 — 컨텍스트 길이 안정화

### 동기
1차 라운드는 컨텍스트를 "8K"까지만 쟀는데, 이는 **벤치 프롬프트 크기 선택**이었지 하드웨어 한계가 아니었다.
Gemma4/Qwen3.6은 네이티브 262,144(256K)를 지원한다. H100 80GB가 실제로 어디까지 감당하는지 규명한다.

### 방법
- **probe** (`probe_maxlen.sh`): 각 (모델×정밀도)를 `--max-model-len 262144 --max-num-seqs 1`로 서빙 →
  성공 시 vLLM의 `GPU KV cache size: N tokens` 리포트, OOM 시 에러의 `estimated maximum model length`를
  직접 읽음(셀당 1~2회 서빙으로 최대치 확정).
- **NIAH** (`niah.py`): "로드 != 작동" 검증. vLLM `/tokenize`로 길이를 정확히 맞춰(추정 아님) 241K 컨텍스트의
  깊이 10/50/90%에 needle을 심고 회수율 측정.

### 결과 — 전 셀 네이티브 256K 도달 (8K는 실한계의 1/32)

| 모델 | 정밀도 | 최대 안정 컨텍스트 | KV 용량(토큰) | NIAH |
|---|---|---|---|---|
| Gemma 26B-A4B (MoE) | bf16 | 256K | 971K | 대표검증 |
| | fp8 | 256K | 1.85M | 3/3 @241K |
| | int8 | 256K | 1.79M | 대표검증 |
| Gemma 31B (dense) | bf16 | 94K -> **256K** (kv-fp8) | 278K(fp8KV) | 3/3 @241K |
| | fp8 | 256K | 405K | 대표검증 |
| | qat-w4a16 | 256K | 519K | 대표검증 |
| Gemma 12B (dense) | bf16 | 256K | 1.96M | 대표검증 |
| | fp8 | 256K | 2.35M | 3/3 @241K |
| Qwen3.6 35B-A3B (MoE + gated-delta) | fp8 | 256K | 1.99M | 3/3 @241K |
| Qwen3.6 27B (gated-delta, dense) | fp8 | 256K | 715K | 3/3 @241K |

*(대표검증 = 동일 아키텍처·KV모드 대표 셀로 NIAH 검증. 데이터: `context/max_context.csv`)*

### 핵심 발견
1. **전 셀 256K 가능.** Gemma는 `sliding_attention`×3 + `full_attention`×1(글로벌 레이어만 전역, `sliding_window=1024`). Qwen3.6는 **두 모델 다** gated-delta 선형어텐션 하이브리드(`linear_attention`×3 + `full_attention`×1, `full_attention_interval=4`) — 전체의 1/4 층만 full-attention(나머지 선형층은 KV≈0, SSM 상태만 유지)이고 그 full 층도 GQA. 두 계열 모두 토큰당 KV가 작아 256K가 여유롭게 들어감.
2. **KV-dtype fp8 = 무거운 bf16의 컨텍스트 레버.** 31B-bf16(62GB)만 auto-KV로 ~94K OOM, `--kv-cache-dtype fp8`로 KV 절반 -> 풀 256K. NIAH로 **fp8 KV가 장거리 회수를 열화 안 시킴** 확인(31B-bf16+kvfp8 3/3@241K).
3. **int8 weight-only는 MoE 전용.** `int8_per_channel_weight_only` 설정이 `linear=None, moe=int8`(vLLM 로그) — 26B-A4B(MoE)는 효과(KV 0.97M->1.79M), dense 12B/31B는 가중치 22.83GiB=bf16로 **no-op**. -> dense 저정밀 축은 fp8+qat가 정답(dense int8은 별도 오프라인 W8A8 필요).
4. **로드 != 작동을 NIAH로 분리 검증.** 메모리만 맞는 게 아니라 241K 전역에서 실제 회수됨(전 대표 셀 3/3, 깊이 10/50/90%).

### 방법론 교훈
- **생성-채점 출력 예산**: 추론/추론서두형 모델(Qwen)은 max_tokens가 작으면 답이 잘려 0점처럼 보임 -> 넉넉히(>=512). NIAH에서 Qwen이 32토큰에 0/3 -> 512토큰에 3/3로 확인.
- **probe 효율화**: 이분탐색 대신 vLLM 자체 KV 리포트/OOM-estimate를 읽어 셀당 1~2회로 최대치 확정.
- **토크나이저 기반 길이 측정**: char/token 추정·재시도 대신 vLLM `/tokenize`로 목표 길이를 정확히 맞춤.

### Phase 3용 고정 설정 (Phase 2 산출)
- 서빙: `--max-model-len`(평가 길이), util 0.95, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
- **31B-bf16/int8**: `--kv-cache-dtype fp8` 필수(아니면 94K cap).
- **Qwen27**: stable `vllm/vllm-openai:v0.22.1` 이미지 + `--max-num-seqs <= 783`(선형어텐션 SSM 상태 캐시 블록 한계 — 27B·35B 공통 메커니즘이나 dense인 27B가 기본 max_num_seqs에서 먼저 도달).
- 생성 채점: max_tokens >= 512(추론형 대응).

---

## 상태 (Phase 3)

다면 평가 배터리(lm-eval/inspect/HRET)는 **thinking 모드 일관성** 재검토 중. 핵심: **Gemma 4·Qwen3.6 둘 다
`enable_thinking` 지원** -> 공정 비교는 전 모델을 동일 설정(no-think 통제 / think 별도)으로 측정해야 함. MMLU-Pro
파일럿(no-think)에서 양자화 무손실(fp8/int8 ~ bf16, qat -0.8%)·크기순(12B<26B<31B) 경향은 확인됨(정본은 일관
프로토콜 재측정 후 확정). 상세는 `PLAN.md`.
