# 재현 가이드

## 1. 환경

| 항목 | 값 |
|---|---|
| GPU | 단일 H100 80GB (GCP Spot, us-central1-a) |
| 런타임 | vLLM docker — `nightly`(AR/MTP·Qwen), `:gemma`(diffusion) |
| 측정 코드 | `../benchmarks/` (`bench_tp.py`, `bench_acc.py`, 러너들) |

이미지 선택 이유: stable 0.22.1은 Gemma4를 `TransformersMultiModalForCausalLM`로 폴백 후 크래시.
nightly만 `gemma4_unified` 네이티브 지원. diffusion은 별도 `:gemma` 이미지.

## 2. 서버 기동 (공통 패턴)

```bash
HF=<your_hf_token>
docker run -d --name gr --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -p 8000:8000 <IMAGE> \
  --model <MODEL> --max-model-len <LEN> --gpu-memory-utilization <UTIL> \
  --no-enable-prefix-caching --host 0.0.0.0 --port 8000 \
  [--quantization <fp8|int8_per_channel_weight_only>] \
  [--speculative-config '<json>']
```

`--no-enable-prefix-caching`는 측정 공정성의 전제(캐시 히트로 인한 거짓 가속 제거).

### 기법별 인자

| 기법 | 추가 인자 |
|---|---|
| AR base | (없음) |
| MTP (Gemma) | `--speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'` |
| MTP (Qwen) | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (임베디드 헤드) |
| MTP (qat) | 드래프터 `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` |
| fp8 | `--quantization fp8` (H100 FP8 텐서코어. A100엔 HW 가속 없음) |
| int8 | `--quantization int8_per_channel_weight_only` (MoE 권장) |
| qat | 모델 `google/gemma-4-31B-it-qat-w4a16-ct` (별도 체크포인트) |
| diffusion | `:gemma` 이미지 + `--max-num-seqs 4` |

## 3. 측정

```bash
# 서버가 :8000/health OK 된 후
python3 benchmarks/bench_tp.py  <label>        # 처리량 short+8K
python3 benchmarks/bench_acc.py acc-<label> 30 # 정확도 N=30
docker rm -f gr                                # 다음 정밀도 전 GPU 정리
```
전 매트릭스 자동 실행은 `benchmarks/run_rem.sh`(31B/12B/diffusion-int8), `run_acc.sh`(정확도).

## 4. 메모리 가이드 (H100 80GB)

| 상황 | 처방 |
|---|---|
| cudagraph가 KV 잠식(util 0.90에서 장문맥 불가) | util 0.95 또는 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` |
| **util 방향** | **올릴수록 KV 큼** (낮추면 OOM 악화 — `04-troubleshooting.md` #8) |
| 31B bf16(62GB) + MTP 드래프터 | 단일 80GB에선 빠듯 → fp8/qat 사용 권장 |
| dense fp8/qat MTP | util 0.90 여유 |
| diffusion | `--max-num-seqs 4` |

## 5. 오케스트레이션 교훈

- 다단계 파이프라인 연쇄는 `flock -n <lockfile>`로 상호배제. 프로세스명 `pgrep` 가드는 self-match로
  데드락 위험(`04-troubleshooting.md` #7).
- 프로세스 정리(`pkill`)는 측정 SSH 채널과 분리(exit 255 방지, #6).
- 스냅샷 VM은 상속된 systemd 유닛이 GPU를 점유할 수 있음 — `nvidia-smi`로 선확인(#2).

## 6. 인프라 여정 (요약)

A100(FP8 HW 없음) → H100 필요 판단 → on-demand STOCKOUT → **Spot 확보**.
쿼터는 `GPUS-PER-GPU-FAMILY-per-project-region`(gpu_family=NVIDIA_H100). 부트 디스크는 모델 캐시 보존을
위해 충분히(500GB) — 온라인 리사이즈는 `growpart`+`resize2fs`.
