# 종합 보고서 — Gemma 4 / Qwen 3.6 추론 벤치마크 (전체 여정)

> **무엇을·왜:** Gemma 4(12B·26B-A4B·31B·diffusiongemma-26B-A4B)와 Qwen 3.6(27B·35B-A3B)을 단일 H100 80GB에서, **정밀도(bf16/fp8/int8/qat) × 추론기법(AR/MTP/diffusion) × 컨텍스트**의 전 조합에 대해 **속도·정확도·메모리** 트레이드오프를 측정했다. 정확도는 thinking-on 동등조건 하에 **3개 프레임워크(lm-eval / inspect_ai / HRET)**로 교차 측정했다. 목적은 "정밀도 우열 비교"가 아니라 **QAT/MTP/Diffusion/thinking 기법이 실제로 쓸만한가 = 속도×정확도×메모리 프론티어**를 규명하는 것.
>
> **핵심 결론 3줄:** ① **양자화(fp8/int8-MoE/qat)는 정확도 무손실** — 속도/메모리만 이득. ② **MTP(speculative decoding)는 정확도 무손실 가속**(n=198 정식 확정, 평균 Δ≈0; dense 2.6× / MoE 1.4~1.8×). ③ **활성 파라미터 적은 MoE가 프론티어를 지배** — dense급 정확도에 3~5× 속도; **diffusion은 4× 최速이나 하드추론·긴문맥 말단에서 약함**.

## 이 보고서 세트 (읽는 순서)

| 파일 | 내용 |
|---|---|
| **README.md**(본 문서) | 초록 · 커버리지 매트릭스 · 데이터 출처 |
| [01-프로토콜.md](01-프로토콜.md) | 실험 여정(A100→H100 라운드1→Phase1·2·3) · 동등조건 배선 · 서빙 구성 · 프레임워크/벤치 · 측정 방법론 |
| [02-시행착오와교훈.md](02-시행착오와교훈.md) | 가설→증거→정정→확정 서사. 답추출 함정 · 서빙 함정 · 프로세스 규율(no-hiding) |
| [03-결과와해석.md](03-결과와해석.md) | 정밀도 sweep · diffusion · MTP(무손실 정식검증) · 컨텍스트 · 4대 발견 · 워크로드 가이드 |
| [04-재현과데이터.md](04-재현과데이터.md) | 환경 · 스크립트 인벤토리 · 재개 패턴 · CSV 스키마/요약 |

## 커버리지 매트릭스 (무엇을·왜 측정/미측정)

정확도 = 3프레임워크 6지표(lm-GPQA · insp-GPQA · MMLU-Pro · IFEval · haerae · KMMLU). ●=전 지표 / ◐=GPQA만 / —=해당없음.

| 모델군 | bf16 | fp8 | int8 | qat | 비고 |
|---|---|---|---|---|---|
| Gemma 12B dense | ● | ● | — | ● | dense int8 = no-op(가중치 bf16) → 미측정 |
| Gemma 26B-A4B MoE | ● | ● | ● | — | qat는 Gemma 12B/31B 전용 |
| Gemma 31B dense | ●(kv-fp8) | ● | — | ● | dense int8 no-op; bf16은 KV fp8 필수 |
| Qwen 35B-A3B MoE | ◐ | ● | ◐ | — | **bf16/int8은 인프라한계(eager 15tok/s)로 GPQA만**(사용자 승인 스코프) |
| Qwen 27B dense-hybrid | ● | ● | ● | — | qat 없음 |
| diffusiongemma 26B-A4B | ● | ● | ● | — | 전용 :gemma 이미지 |

**기법 축:**
- **MTP**: 9셀(12B/26B/31B/qwen35/qwen27 정밀도별) speedup + **GPQA full 198 정확도 직접 실측** + 토큰동일성 검증(대조군 포함).
- **컨텍스트(NIAH)**: Gemma/Qwen 전 대표셀 **3/3 @ ~241K**; diffusion **2/3 @ 32~48K**(말단 약점).

**모든 "모델"은 대표 정밀도에서 3프레임워크 전부 커버됨.** 유일한 부분커버는 Qwen35 bf16/int8 — 70GB가 단일 H100에서 CUDA graph를 못 올려 enforce-eager(15 tok/s)로 폴백되는 **인프라 한계** 때문이며, 양자화 무손실만 GPQA로 확인하면 충분하다는 판단으로 사용자가 GPQA-only 스코프를 승인했다(인위적 제약 아님).

## 데이터 출처

- **마스터 CSV**: `../results_consolidated.csv` (191행) — 스키마 `ts,cell,framework,benchmark,limit,metric,score`. 본 보고서의 모든 수치 근거.
- **컨텍스트**: `../context/max_context.csv`, `../context/FINDINGS.md`.
- **초기 라운드 원본(아카이브)**: `../legacy/`(A100~), `../legacy/h100-round1/`(1차 H100), `../legacy/WORKLOG.md`.
- **재현 스크립트**: `../evals/`, `../context/niah.py`, `../bench_tp.py`.

> 환경: 단일 H100 80GB (GCP Spot, us-central1-a), vLLM docker. 측정 기간 2026-06-11 ~ 06-17.
