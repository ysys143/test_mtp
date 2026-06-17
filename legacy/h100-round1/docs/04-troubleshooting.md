# 트러블슈팅

각 항목: **증상 → 오진 → 진짜 원인 → 해법 → 교훈.** ([세션] = 재구조화 시점에 발생/규명)

## 1. [치명적] streaming 측정이 spec decode를 ~3배 과소측정
- **증상:** acceptance 100%인데 streaming으로는 MTP 가속이 안 보임. Qwen은 "0.44x 손해"까지.
- **오진:** "MoE라 불리하다", "검증 비용이 이득을 먹는다".
- **진짜 원인:** streaming `decode_tok/s`는 spec decode의 버스트 커밋을 과소측정. (`02-methodology.md` §1)
- **해법:** non-streaming 총처리량으로 전면 재측정 → "손해"가 2.6~3.0x 이득으로 반전.
- **교훈:** spec decode/MTP는 절대 streaming 토큰율로 재지 말 것. acceptance length로 교차검증.

## 2. 유령 systemd 서비스가 GPU를 훔침
- **증상:** "H100에서 31B/fp16이 메모리 부족으로 안 돈다."
- **오진:** 모델이 너무 크다 / H100 설정 문제.
- **진짜 원인:** A100 스냅샷에 따라온 `ops-llm-test.service`가 부팅 시 호스트 vLLM 자동 기동 → GPU ~33GB 상시 점유.
- **해법:** `sudo systemctl disable ops-llm-test` + util 0.95. (31B는 H100에서 정상.)
- **교훈:** 스냅샷 VM은 상속된 systemd 유닛을 의심. `nvidia-smi`로 "내가 안 띄운 프로세스"부터 확인.

## 3. diffusion 콜드스타트 혼동
- **증상:** diffusion이 82.9 tok/s로 느리게 측정.
- **진짜 원인:** non-streaming 측정에 콜드스타트(첫 컴파일/워밍업)가 섞임.
- **해법:** warmup 추가 후 재측정 → 순수 generation 615(bf16)/864(fp8) tok/s.
- **교훈:** diffusion은 prefill/decode 경계가 없어 워밍업 필수.

## 4. [해결] Qwen 27B "비호환"은 사실 설정 문제 (Mamba-하이브리드)
- **증상:** Qwen 27B-FP8 base가 nightly에선 enforce-eager로 14 tok/s, v0.22.1에선 DIED. MTP도 "FAILED".
- **오진:** "vLLM nightly 비호환 / 측정 불가"로 단정(반복).
- **진짜 원인:** (1) Qwen3.6-27B은 **Mamba-하이브리드(SSM+attention)** — 디코드 시퀀스당 Mamba 캐시 블록
  필요. 기본 `max_num_seqs=1024`가 가용 블록(783) 초과 → `ValueError: ... exceeds available Mamba cache
  blocks` → cudagraph 캡처 실패 DIED. (2) MTP는 init이 느림(드래프터+mamba/attention page 정렬+cudagraph,
  ~366s) → 헬스 타임아웃(360s)이 짧아 "FAILED"로 오판.
- **해법:** stable `v0.22.1` 이미지 + `--max-num-seqs 256` + 넉넉한 헬스 타임아웃 → base **79.2 tok/s(90%)**,
  MTP **146.9(1.85x)** 정상 측정.
- **교훈:** 새 아키텍처(Mamba-하이브리드)의 "안 뜸"을 비호환으로 단정 말 것. 에러를 끝까지 캡처하면 대개
  한 플래그 문제. **이 세션 Qwen "실패" 3건이 전부 비본질적**(max_tokens 잘림 #10 → max_num_seqs 초과 →
  헬스 타임아웃 부족).

## 5. 디스크 churn / 온라인 리사이즈
- **증상:** 50GB 모델을 지웠다 다시 받기 반복.
- **해법:** 부트 디스크 194GB→500GB 온라인 리사이즈(`growpart` + `resize2fs`).
- **교훈:** 모델 캐시(`~/.cache/huggingface`)를 보존할 디스크부터 확보.

## 6. SSH exit 255 wedge
- **증상:** 같은 SSH 명령 안에서 `pkill -9 VLLM::EngineCore` 실행 시 세션이 끊김(255).
- **진짜 원인:** pkill이 부모 SSH 세션까지 영향(별개로 CPU 포화→sshd wedge 사례도 있음).
- **해법:** pkill은 별도/detached 명령으로 분리.
- **교훈:** 프로세스 정리를 측정 명령과 같은 SSH 채널에 섞지 말 것.

## 7. [세션] pgrep self-match 데드락 — 파이프라인 정지
- **증상:** ACC 완료 후 `run_rem`이 시작 안 됨. GPU ~2분 idle, `run_rem.out` 파일조차 없음.
- **진짜 원인:** 체인 waiter 가드 `if ! pgrep -f run_rem.sh`에서 **waiter 자신의 커맨드라인에 "run_rem.sh"가
  포함**돼 pgrep이 자기 자신을 매치 → 가드 항상 false → run_rem 영영 미실행 → 없는 파일 무한 폴링.
- **해법:** 데드락 waiter kill → `flock -n <lockfile> bash run_rem.sh`로 재기동(inode 락, 이름 self-match 면역).
- **교훈:** 프로세스 존재 검사를 모니터 자신의 cmdline에 등장하는 이름으로 하지 말 것. 상호배제는 `flock`.

## 8. [세션] 31B-bf16-MTP OOM + util을 거꾸로 내린 실수
- **증상:** 31B-bf16-mtp가 util 0.95에서 OOM.
- **실수:** 재측정 util을 **0.88로 낮춤** → 더 악화(`estimated max len 1952`, KV 1.65GB).
- **진짜 원인:** `gpu_memory_utilization`은 vLLM 점유 상한 — 낮추면 KV가 더 작아짐. 게다가 62GB 가중치가
  80GB의 78%를 먹어 드래프터+cudagraph 동시 수용 시 KV<6.89GB(8192토큰 필요)로 부족.
- **해법/결론:** util 0.97 + `--enforce-eager`로만 기동되나 그 수치(21.4/58.9)는 비교 불가 →
  **단일 H100 80GB에서 깨끗한 측정 불가**로 결론. fp8/qat에선 MTP 정상.
- **교훈:** util 방향 직관("올려야 KV가 큼"). bf16 dense 31B는 H100 80GB에서 spec decode 경계.

## 9. [세션] enforce-eager 아티팩트 식별
- **증상:** 31B-bf16-mtp short 21.4 < 8K 58.9 (역전).
- **진짜 원인:** enforce-eager(no CUDA graph) + MTP 다중 forward → short에서 커널런치 오버헤드 지배.
- **해법:** 해당 수치는 다른 MTP와 비교 불가로 각주 처리(매트릭스에 OOM 표기).
- **교훈:** enforce-eager 수치는 cudagraph 수치와 같은 표에 섞지 말 것.

## 10. [해결] Qwen 정확도 6.7% → 83.3% (채점 토큰예산 아티팩트)
- **증상:** Qwen35-A3B-FP8 정확도 2/30=6.7% (Gemma 전 모델 100%).
- **진짜 원인:** `bench_acc.py`의 채점 호출이 **`max_tokens=24`**. Qwen은 답 앞에 추론 서두를 붙이는데
  24토큰 예산이 서비스명 출력 전에 소진 → 잘린 서두만 채점됨(substring 불일치). Gemma는 이름만 바로 출력.
- **해법:** `bench_acc2`(max_tokens 24→256 + "마지막 언급 서비스명" 추출). Gemma 26B-fp8 control로 100%
  재확인(공정성) → Qwen35 **83.3%**, Qwen27 **90.0%** 실측. 옛 채점기 대조값 6.7% 유지(아티팩트 확정).
- **교훈:** 멀티모델 정확도 비교 시 토큰예산·답 추출기를 모델 출력 포맷에 강건하게. 생성-파싱 채점보다
  loglikelihood 채점(lm-eval)이 이런 아티팩트에 구조적으로 강함.

---
> 이 트러블슈팅이 만든 측정 원칙은 `02-methodology.md`, 재현 시 주의점은 `05-reproduce.md`.
