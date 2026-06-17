# benchmarks/

H100 측정에 사용한 코드. `opscheck@<H100-VM>`에서 회수(2026-06-13).

## 핵심 측정기
| 파일 | 역할 |
|---|---|
| `bench_tp.py` | **처리량**. non-streaming, 출력 512토큰 고정(`ignore_eos`), `total_tp = 출력토큰/전체latency`. warmup 1회(cudagraph) 후 short + 8K 두 컨텍스트 측정. **streaming 금지가 핵심**(`../docs/02-methodology.md` 참조). |
| `bench_acc.py` | **정확도**. ops 근본원인 식별 — 한 서비스가 먼저 죽고(line 40) 의존 서비스가 cascade(upstream timeout)되는 합성 인시던트 로그에서 근본원인을 보기 중 선택, ground-truth substring 채점, N=30. **주의: `max_tokens=24`라 추론 서두를 붙이는 모델(Qwen)은 답이 잘림** → `../docs/04-troubleshooting.md` #10. |

## 매트릭스 러너
| 파일 | 역할 |
|---|---|
| `run_rem.sh` | 메인 매트릭스: diffusion-int8 + 31B(bf16/fp8/qat × base/mtp) + 12B(bf16/fp8 × base/mtp). serve→bench_tp(+bench_acc)→clean_gpu 순차. flock로 단독 실행 권장. |
| `run_acc.sh` | 정확도 전용: 26B(bf16/fp8/int8) + diffusion(bf16/fp8) + Qwen35. |
| `qwen_diag.sh` | Qwen35-A3B MTP(γ1/γ2) + Qwen27 호환성 진단. |
| `run_fix.sh`, `run_fix2.sh` | 31B-bf16-MTP 재시도(메모리 제약 규명, `docs/04` #8). |

## 사용
```bash
# 서버는 docker로 별도 기동 (../docs/05-reproduce.md). 서버가 :8000에 떠 있는 상태에서:
python3 bench_tp.py <label>          # 처리량 short+8K
python3 bench_acc.py acc-<label> 30  # 정확도 N=30
```
