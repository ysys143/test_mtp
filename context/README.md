# context/ — Phase 2: 컨텍스트 길이 안정화

목표: 각 (모델 × 정밀도)에서 H100 80GB에 **안정 서빙되는 최대 컨텍스트**를 규명·고정한다.
"로드된다"가 아니라 "그 길이에서 실제로 정확히 동작한다"까지 검증. 상세 방법은 [../PLAN.md](../PLAN.md) Phase 2.

## 예정 산출물
| 파일 | 역할 |
|---|---|
| `probe_maxlen.sh` | 셀별 `--max-model-len` 이분 탐색(OOM 직전 최대치) |
| `niah.py` | needle-in-a-haystack 검증 (로드 ≠ 작동) |
| `max_context.csv` | (model, precision, max_stable_ctx, kv_dtype, util, niah_pass) 고정값 |

## 레버 (컨텍스트 최대화)
저정밀 가중치(fp8/int8/qat) · `--kv-cache-dtype fp8`(~2x) · util 0.95 ·
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` · `--enable-chunked-prefill` · (네이티브 초과 시) rope-scaling.

> 측정 코드 재사용: `../legacy/h100-round1/benchmarks/`의 `bench_tp.py` 등 참고.
