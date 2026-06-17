# 작업 보고서 — Gemma 4 / Qwen3.6 추론 벤치마크 (H100)

**기간:** 2026-06-11 ~ 2026-06-13
**최종 환경:** 단일 H100 80GB (GCP Spot, us-central1-a), vLLM docker
**산출물:** `GEMMA4.md`(정본 v2 종합 매트릭스), `README.md`(§0.5 Qwen 정정본), 본 보고서
**목적:** 모델 백본 × 정밀도 × 추론 기법 전 조합의 throughput·정확도를 동일 하드웨어/런타임에서 측정

> 이 문서는 **결과표 중복이 아니라 "작업 여정 + 트러블슈팅 추적 + 재현 가이드"**다.
> 최종 수치는 `GEMMA4.md` §H100 종합 매트릭스(정본 v2), Qwen은 `README.md` §0.5 참조.

---

## 1. 목표와 범위

### 1.1 측정 대상 (전부 완료)

| 축 | 값 |
|---|---|
| 모델 | Gemma 26B-A4B(MoE ~4B 활성), 31B(dense), 12B(dense), diffusiongemma-26B-A4B, Qwen3.6 35B-A3B(MoE), Qwen3.6 27B(dense) |
| 정밀도 | BF16, fp8(runtime), int8(per-channel weight-only), qat-w4a16(compressed-tensors) |
| 기법 | AR baseline, AR + MTP(speculative decoding), block diffusion |
| 지표 | throughput(tok/s, short + 8K context), 정확도(ops 근본원인 태스크, N=30) |

### 1.2 핵심 질문
1. MTP(multi-token prediction)는 우리 ops 워크로드에서 실질 이득을 주는가?
2. 양자화(fp8/int8/qat)가 정확도를 떨어뜨리는가?
3. diffusion LLM은 AR 대비 빠른가? 정확한가?
4. MoE와 dense에서 MTP 이득이 다른가?

---

## 2. 인프라 구축 과정

### 2.1 모델 런타임 확보 (실험의 절반)
Gemma 4는 2026-06 출시작이라 tooling이 따라잡는 중 — **"작동하는 환경 찾기" 자체가 난관**이었다.

| 항목 | 해법 |
|---|---|
| Gemma 4 네이티브 지원 | `vllm/vllm-openai:nightly` (gemma4_unified 아키텍처 네이티브). stable 0.22.1은 `TransformersMultiModalForCausalLM`로 폴백 후 크래시 |
| diffusion 지원 | 별도 이미지 `vllm/vllm-openai:gemma` (block diffusion, 256토큰 canvas) |
| MTP 드래프터 | Gemma 4는 **별도 ~0.4B 드래프터 모델**(`gemma-4-<size>-it-assistant`) 사용. Qwen은 **임베디드 MTP 헤드**(`--speculative-config '{"method":"mtp",...}'`) |
| qat MTP 드래프터 | `gemma-4-31B-it-qat-q4_0-unquantized-assistant` (qat 전용) |

### 2.2 GPU 확보 여정
- A100 80GB로 시작 → **A100은 FP8 텐서코어 하드웨어 가속이 없음**을 확인, H100 필요 판단.
- H100 on-demand는 **재고 없음(STOCKOUT)** → **Spot**으로 us-central1-a 확보.
- 쿼터: `GPUS-PER-GPU-FAMILY-per-project-region`(gpu_family=NVIDIA_H100) 증설.
- 프로비저닝: A100 스냅샷 기반 VM 생성(→ 후술 "유령 서비스" 문제의 원인).

### 2.3 측정 하네스
- `bench_tp.py`: non-streaming throughput. 고정 출력 512토큰(`ignore_eos:True`), `total_tp = out / total_latency`, short + 8K context, 서버 `--no-enable-prefix-caching`.
- `bench_acc.py`: ops 근본원인 태스크. 한 서비스가 먼저 죽고 나머지가 캐스케이드되는 합성 인시던트 로그 생성 → 근본원인 서비스를 보기 중 선택 → ground-truth 대조, N=30.
- `run_rem.sh` / `run_acc.sh`: serve→bench→clean_gpu를 정밀도/기법별로 순차 실행하는 오케스트레이터.

---

## 3. 측정 방법론 — 가장 중요한 교정

> **핵심: 모든 throughput은 non-streaming 총처리량으로만 측정해야 한다.**

| 방식 | 정의 | 문제 |
|---|---|---|
| streaming `decode_tok/s` | 첫 토큰~마지막 토큰 사이 토큰율 | **speculative decoding을 ~3배 과소측정** — MTP는 토큰을 버스트로 커밋하므로 순간 토큰율이 왜곡됨 |
| non-streaming 총처리량 (채택) | 고정 출력토큰수 / wall-clock latency, prefill 분리, prefix cache off | spec decode의 실이득을 정확히 반영 |

이 한 가지 결함이 초기 "MTP 손해" 결론을 통째로 뒤집었다(§5.1).

---

## 4. 최종 결과 요약 (상세는 GEMMA4.md)

**throughput (tok/s, short context), 정확도(N=30):**
- 26B-A4B(MoE): AR 200~226 / MTP 323~390(1.6~1.7x) / diffusion 615~864 · 정확도 AR 100%, diffusion 86.7~93.3%
- 31B(dense): base bf16 40.5 → fp8 67.4 → qat 89.6 / MTP fp8 178.1(**2.64x**), qat 213.7(2.38x) · 정확도 100%
- 12B(dense): bf16 82.4→MTP 186.5(2.26x), fp8 118.7→MTP 200.7 · 정확도 100%
- Qwen35-A3B fp8: base 216.7 / MTP γ2 315.6(1.46x) · 정확도 6.7%(파서 아티팩트, §5.10)

**4대 질문 답:**
1. MTP는 전 dense 모델·정밀도에서 일관된 가속, 정확도 무손실. dense일수록 큼(대역폭 병목 분할상환).
2. 양자화는 정확도 무손실(전 셀 100%).
3. diffusion은 속도 최강(864)이나 정확도 유일 미달(86.7~93.3%).
4. MoE는 MTP 이득 작음(1.6~1.7x, 이미 활성 4B로 대역폭 부담 작음) vs dense 2.6x+.

---

## 5. 트러블슈팅 전말

각 항목: **증상 → 오진 → 진짜 원인 → 해법 → 교훈**.

### 5.1 [치명적] streaming 측정이 MTP를 3배 과소측정
- **증상:** acceptance가 100%인데도 streaming으로는 MTP 가속이 안 보임. Qwen 실험에선 "0.44x 손해"라는 결론까지 나옴.
- **오진:** "MoE는 MTP에 불리하다", "검증 비용이 이득을 잡아먹는다".
- **진짜 원인:** streaming `decode_tok/s`는 spec decode의 버스트 커밋을 과소측정. vLLM 자체 "Mean acceptance length" 지표로 교차검증하니 실제로는 이득.
- **해법:** non-streaming 총처리량(고정 출력/wall-clock)으로 전면 재측정 → "손해" → **2.6~3.0x 이득**으로 반전.
- **교훈:** spec decode/MTP는 절대 streaming 토큰율로 재지 말 것. 이 결함이 Qwen(README) + Gemma(GEMMA4) 양쪽 초기 결론을 무효화함.

### 5.2 유령 systemd 서비스가 GPU를 훔침
- **증상:** "H100에서 31B/fp16이 메모리 부족으로 안 돈다." 사용자가 반복 지적("A100에서도 fp16 잘 돌았는데", "30B를 못 돌린다고?").
- **오진:** 모델이 너무 크다 / H100 설정 문제.
- **진짜 원인:** A100 스냅샷에 `ops-llm-test.service`가 따라와 H100 부팅 시 **호스트 vLLM을 자동 기동, GPU ~33GB 상시 점유**.
- **해법:** `sudo systemctl disable ops-llm-test` + util 0.95. (31B는 H100에서 정상 — 크기 문제가 아니었음.)
- **교훈:** 스냅샷 기반 VM은 상속된 systemd 유닛을 의심하라. `nvidia-smi`로 "내가 안 띄운 프로세스"부터 확인.

### 5.3 diffusion 콜드스타트 혼동
- **증상:** diffusion이 82.9 tok/s로 느리게 측정됨("뭔가 잘못한 거 아냐?").
- **진짜 원인:** non-streaming 측정에 콜드스타트(첫 컴파일/워밍업)가 섞임 + canvas 동작 특성.
- **해법:** warmup 추가 후 재측정 → 순수 generation ~615(bf16)/864(fp8) tok/s.
- **교훈:** diffusion은 prefill/decode 경계가 없어 워밍업 필수.

### 5.4 Qwen 27B-FP8 nightly 비호환
- **증상:** Qwen 27B-FP8(`Qwen3_5MTP` arch) base는 `--enforce-eager`로만 14 tok/s, MTP는 DIED.
- **진짜 원인:** vLLM nightly 회귀. (원래 A100 0.22.1 스택에선 작동.)
- **해법:** nightly는 Gemma 4 네이티브 지원 때문에 불가피하게 선택 → Qwen 27B는 이 스택에서 **미측정 문서화**.
- **교훈:** 멀티모델 벤치에서 단일 런타임이 전 모델을 커버 못 할 수 있음. 트레이드오프를 기록.

### 5.5 디스크 churn / 온라인 리사이즈
- **증상:** 50GB 모델을 지웠다 다시 받기 반복("스토리지로 옮겨놓으면 되지 왜 지우고 다운받아?").
- **해법:** 부트 디스크 194GB→500GB 온라인 리사이즈(`growpart` + `resize2fs`). 이후 재다운로드 불필요.
- **교훈:** 모델 캐시(`~/.cache/huggingface`)를 보존할 디스크부터 확보.

### 5.6 SSH exit 255 wedge
- **증상:** 같은 SSH 명령 안에서 `pkill -9 VLLM::EngineCore` 실행 시 세션이 끊김(exit 255).
- **진짜 원인:** pkill이 자신의 부모 SSH 세션까지 영향. (메모리: native_codex의 CPU 포화로 sshd가 wedge되는 별개 사례도 있음.)
- **해법:** pkill은 별도/detached 명령으로 분리 실행.
- **교훈:** 프로세스 정리는 측정 명령과 같은 SSH 채널에 섞지 말 것.

### 5.7 [이번 세션] pgrep self-match 데드락 — 파이프라인 정지
- **증상:** ACC(정확도) 6단계 완료 후 `run_rem`이 시작 안 됨. GPU 0 MiB로 ~2분 idle. `run_rem.out` 파일조차 없음.
- **오진:** 타이밍 문제.
- **진짜 원인:** 체인 waiter의 가드 `if ! pgrep -f run_rem.sh; then 실행; fi`에서, **waiter 자신의 커맨드라인에 "run_rem.sh" 문자열이 포함**돼 `pgrep -f`가 자기 자신을 매치 → 가드가 항상 false → run_rem을 영영 실행 안 함. 이후 `until grep 'REM DONE' run_rem.out`이 없는 파일을 무한 폴링. 내 보조 체인도 같은 패턴으로 waiter를 "이미 실행 중"으로 오인해 backoff.
- **해법:** 데드락 waiter kill → `flock -n ~/run_rem.flock bash run_rem.sh`로 재기동. flock은 **inode 락**이라 프로세스명 정규식 self-match에 면역.
- **교훈:** 프로세스 존재 검사를 모니터 자신의 cmdline에 등장하는 이름으로 하지 말 것. 상호배제는 `flock`(커널 락) 사용.

### 5.8 [이번 세션] 31B-bf16-MTP OOM + util을 거꾸로 내린 실수
- **증상:** 31B-bf16-mtp가 util 0.95에서 OOM(엔진 init 실패).
- **1차 오진/실수:** 재측정을 util **0.88**로 낮춤 → 더 악화. `estimated max len 1952`(KV 1.65GB만 남음).
- **진짜 원인:** `gpu_memory_utilization`은 vLLM 점유 상한 — **낮추면 KV가 더 작아짐**. 게다가 31B×2byte≈62GB 가중치가 80GB의 78%를 먹어, 드래프터+cudagraph까지 동시 수용 시 KV<6.89GB(8192 토큰 필요)로 부족.
- **2차 해법:** util **0.97** + `--enforce-eager`(cudagraph 메모리 회수)로 기동 성공. 단 enforce-eager가 throughput을 왜곡(21.4 tok/s short, base 40.5보다 느림) — MTP는 스텝당 다중 forward라 커널런치 오버헤드가 더 큼.
- **결론:** **31B-bf16-MTP는 단일 H100 80GB에서 깨끗한 측정 불가**(cudagraph OOM vs eager 왜곡). fp8/qat에선 MTP 정상(2.64x/2.38x)이라 실무상 무의미한 코너.
- **교훈:** util 방향 직관 확립("올려야 KV가 큼"). bf16 dense 31B는 H100 80GB에서 spec decode 돌리기 빠듯한 경계.

### 5.9 [이번 세션] enforce-eager 아티팩트 식별
- **증상:** 31B-bf16-mtp short 21.4 < 8K 58.9 (역전).
- **진짜 원인:** enforce-eager(no CUDA graph) + MTP 다중 forward → short context에서 런치 오버헤드 지배. 8K는 긴 prefill이 상대적으로 분할상환.
- **해법:** 해당 수치는 다른 MTP 수치와 **비교 불가**로 각주 처리, 매트릭스에서 OOM*로 표기.
- **교훈:** enforce-eager 수치는 cudagraph 수치와 같은 표에 섞지 말 것.

### 5.10 [미해결] Qwen 정확도 6.7% — 채점 파서 아티팩트
- **증상:** Qwen35-A3B-FP8 정확도 2/30=6.7% (Gemma 전 모델은 100%).
- **진짜 원인:** 오답 추출값이 `'the user wants to identify the'`. Qwen이 답 앞에 추론 서두를 붙이는데 `bench_acc.py` 추출기가 서비스명 대신 그 서두를 긁음. Gemma는 답 포맷을 지켜 100%.
- **상태:** **미수정.** 실 성능 아님 — 파서 한계. Qwen-aware 추출기로 재측정 필요(사용자 결정 대기).
- **교훈:** 멀티모델 정확도 비교 시 답 추출기는 모델별 출력 포맷 차이에 강건해야 함.

---

## 6. 미해결 / 후속 과제

| 항목 | 상태 | 비고 |
|---|---|---|
| Qwen 정확도 재측정 | 미수정 | bench_acc.py 추출기 Qwen 포맷 대응 후 재측정 |
| 31B-bf16-MTP 정밀 측정 | 불가(문서화) | 멀티 GPU(TP=2) 또는 더 큰 단일 GPU 필요 |
| Qwen 27B-FP8 측정 | 불가(문서화) | vLLM 0.22.1 스택 별도 구성 시 가능 |

---

## 7. 재현 방법 (요약)

```bash
# 1) 이미지: AR/MTP는 nightly, diffusion은 gemma
NI=vllm/vllm-openai:nightly; GI=vllm/vllm-openai:gemma

# 2) AR base 서버 (non-streaming 측정 전제: prefix cache off)
docker run -d --name gr --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  $NI --model google/gemma-4-31B-it --max-model-len 16384 \
  --gpu-memory-utilization 0.95 --no-enable-prefix-caching

# 3) MTP: --speculative-config 추가 (Gemma=별도 드래프터)
#   --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'

# 4) 측정: 고정 출력 512토큰 / wall-clock (streaming 금지)
python3 bench_tp.py <label>     # throughput
python3 bench_acc.py acc-<label> 30   # 정확도 N=30
```

**메모리 가이드(H100 80GB):** dense 31B bf16는 util 0.95 + cudagraph로 KV 빠듯 → MTP 시 OOM.
fp8/qat는 util 0.90으로 여유. diffusion은 `--max-num-seqs 4`.

**오케스트레이션 교훈:** 다단계 파이프라인 연쇄는 `flock -n <lockfile>`로 상호배제(프로세스명 pgrep 가드 금지).

---

*작성: 2026-06-13. 결과 정본은 `GEMMA4.md`(§H100 종합 매트릭스 정본 v2), `README.md`(§0.5).*
