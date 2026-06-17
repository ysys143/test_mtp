# Gemma 4 / Qwen3.6 추론 벤치마크 (H100)

단일 H100 80GB에서 **모델 백본 × 정밀도 × 추론 기법**의 처리량·정확도를 동일 런타임(vLLM)으로 측정한 기록.

## 핵심 결론

1. **MTP(speculative decoding)는 dense 모델에서 2.6x 가속 + 정확도 무손실** — dense일수록 큼(31B fp8 2.64x, MoE는 1.5~1.7x).
2. **양자화(fp8/int8/qat)는 정확도를 떨어뜨리지 않음** — 전 셀 100%, 속도만 단조 상승. (단 이 ops 태스크는 천장 포화 → "현저한 손상 없음"의 하한 보증; [01-results](docs/01-results.md) 천장 효과 주의)
3. **diffusion이 속도 최강(864 tok/s)이나 정확도는 유일하게 100% 미달**(86.7~93.3%, int8 최고).
4. **속도-정확도 동시 최적 = Gemma 26B-A4B-fp8** — MoE의 적은 활성 파라미터가 처리량·메모리 양쪽에 유리.

## 결과 하이라이트 (tok/s, short context / 정확도 N=30)

| 모델 | 정밀도 | AR | +MTP | diffusion | 정확도 |
|---|---|---|---|---|---|
| Gemma 26B-A4B (MoE) | fp8 | 226 | 390 | **864** | 100% (diff 90%) |
| Gemma 31B (dense) | fp8 | 67 | **178 (2.64x)** | — | 100% |
| Gemma 31B (dense) | qat | 90 | 214 | — | 100% |
| Gemma 12B (dense) | bf16 | 82 | 187 (2.26x) | — | 100% |
| Qwen3.6 35B-A3B (MoE) | fp8 | 217 | 316 (1.46x) | — | 83.3% |
| Qwen3.6 27B (Mamba-hybrid) | fp8 | 79 | 147 (1.85x) | — | 90.0% |

전체 매트릭스: **[docs/01-results.md](docs/01-results.md)**.
Qwen 정확도는 수정 채점기(`bench_acc2`)로 측정 — 초기 6.7%는 토큰예산 아티팩트였음([docs/04 #10](docs/04-troubleshooting.md)).

## 문서 맵

| 문서 | 내용 |
|---|---|
| [docs/01-results.md](docs/01-results.md) | 정본 종합 매트릭스 (속도 + 정확도, 불가/예외 셀) |
| [docs/02-methodology.md](docs/02-methodology.md) | non-streaming 측정 원리, acceptance-length 검증, 통제, 정확도 태스크 |
| [docs/03-findings.md](docs/03-findings.md) | 해석: MTP / 양자화 / diffusion / MoE↔dense |
| [docs/04-troubleshooting.md](docs/04-troubleshooting.md) | 트러블슈팅 10건 (증상→원인→해법→교훈) |
| [docs/05-reproduce.md](docs/05-reproduce.md) | 재현 가이드 (docker, 메모리, 인프라) |
| [benchmarks/](benchmarks/) | 측정 코드 (`bench_tp.py`, `bench_acc.py`, 러너) |
| [data/results.csv](data/results.csv) | 기계가독 매트릭스 |
| [legacy/](legacy/) | 초기 보고서·로그·스크립트 아카이브(정정 이력 포함) |

## 환경

단일 H100 80GB (GCP Spot) · vLLM `nightly`(AR/MTP·Qwen) / `:gemma`(diffusion) · 측정 2026-06-11~13.

> **읽는 순서 제안:** 결론(이 페이지) → 결과표(01) → 왜 믿을 수 있나(02) → 해석(03).
> 측정이 왜 어려웠는지는 04, 직접 돌리려면 05.
