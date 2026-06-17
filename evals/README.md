# evals/ — Phase 3: 다면 평가 배터리

**전제: Phase 2(컨텍스트 안정화)가 완료된 후에만 착수.** Phase 2에서 고정한 컨텍스트를 사용한다.

세 도구를 같은 vLLM OpenAI 엔드포인트에 붙인다. 상세는 [../PLAN.md](../PLAN.md) Phase 3.

| 하위 | 축 | 대표 벤치 |
|---|---|---|
| `lm_eval/` | 영어 추론·장문맥 | GPQA-Diamond, MMLU-Pro, BBH, RULER |
| `inspect/` | tool / 에이전트 / 난도 상향 자작 ops | BFCL·τ-bench, custom |
| `hret/` | 한국어 | KMMLU, HAE-RAE Bench, HRM8K |

원칙: temp 0·프롬프트 고정·동일 컨텍스트. 정밀도×기법 축 유지. 한·영 점수 직접 비교 금지.
정확도 채점은 가능하면 loglikelihood 방식(생성-파싱 아티팩트 회피, `legacy/h100-round1` 교훈).
