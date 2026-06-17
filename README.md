# LLM 추론 평가 (Gemma 4 / Qwen3.6, H100)

> **완료. 정본 = 전체 여정 종합 보고서 → [report/README.md](report/README.md)** (프로토콜·시행착오·결과·재현 5파일). 아래 단편 문서(REPORT/MATRIX/RESULTS_PHASE3 등)는 종합본으로 대체됨(원본 보존).

## 구조
| 경로 | 내용 |
|---|---|
| **[report/](report/README.md)** | **★ 종합 보고서(정본)** — 01프로토콜·02시행착오·03결과·04재현 |
| [results_consolidated.csv](results_consolidated.csv) | 마스터 데이터 191행 (전 표의 근거) |
| [REPORT.md](REPORT.md) · [MATRIX.md](MATRIX.md) · [RESULTS_PHASE3.md](RESULTS_PHASE3.md) | (대체됨) Phase별 단편 원본 |
| [PLAN.md](PLAN.md) | 단계별 작업 계획 (Phase 0~3) |
| `context/` | **Phase 2(완료)** — (모델×정밀도)별 최대 안정 컨텍스트 규명·고정 ([FINDINGS](context/FINDINGS.md)) |
| `evals/` | **Phase 3** — 다면 평가 배터리 (lm-eval / inspect_ai / HRET) |
| `legacy/h100-round1/` | 1차 H100 라운드(속도·정확도 매트릭스 + 트러블슈팅) — 아카이브 |
| `legacy/` (그 외) | 초기 A100 라운드 보고서·로그 |

## 이전 라운드 요약 (legacy/h100-round1)
- non-streaming 측정으로 MTP 2.6x(dense)·무손실 확인, 양자화 정확도 무손실, diffusion 속도 최강(864 tok/s).
- 한계: 측정 컨텍스트가 8K에 머물렀고(하드웨어 한계 아님), 정확도 태스크가 N=30·천장 포화로 변별력 부족.
- → 본 라운드는 (1) 컨텍스트 실한계 규명, (2) 표준 배터리로 평가를 재구성한다.
