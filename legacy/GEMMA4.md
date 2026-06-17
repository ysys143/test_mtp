# Gemma 4 변종 비교 실험 보고서 (original / QAT / MTP / diffusion)

**대상:** Google Gemma 4 패밀리 — 26B-A4B (MoE) / 31B (Dense)
**환경:** 단일 A100 80GB, vLLM (docker), GCP
**측정일:** 2026-06-11
**워크로드:** SRE 인시던트 분석 모사 — 8K 및 128K(~123K 토큰) 컨텍스트, 변종별 max 동시성

---

> [!IMPORTANT]
> **측정 방법 정정 (2026-06-13, H100 재측정).** 이 보고서의 **A100 MTP 수치는 streaming
> 측정 아티팩트로 무효**다. streaming `decode_tok/s`(첫토큰~마지막토큰 토큰율)는 speculative
> decoding이 토큰을 **버스트로 커밋**하는 것을 과소측정한다(약 3배). non-streaming(고정 출력
> 토큰수 / wall-clock, prefill 분리)으로 H100에서 재측정한 결과 **MTP는 전 정밀도에서 2.6~3.0x
> 이득**이다. 아래 §0, §H100을 정본으로 삼고, §3~5의 A100 MTP "손해" 서술은 폐기한다.
> (같은 결함이 선행 Qwen MTP 실험(README.md)의 "0.44x 손해" 결론도 무효화한다 — 재측정 대기.)

## 0. 한 줄 결론 (정정본)

> **H100 단일 GPU, non-streaming 정밀 측정 결과: (1) Gemma 4 MTP는 BF16/fp8/qat 전 정밀도에서
> 2.6~3.0x throughput 이득 — 제3자 벤치(JarvisLabs, vLLM 클라우드 제공사) 3.11x와 일치. (2) acceptance는 워크로드
> 의존(예측가능 91~100%, 에세이 63%)이며 정상. (3) diffusion은 A100에선 동시성/128K 크래시지만
> H100에선 정상 작동(순수 generation ~836 tok/s). (4) dense QAT(w4a16)는 메모리·속도·feasibility
> 강점 유지. (5) fp8(런타임)은 H100 FP8 텐서코어로 A100(weight-only) 대비 1.78x.**

---

## H100 결과 (정본, non-streaming, 31B)

| target | baseline tok/s | + MTP tok/s | speedup | acceptance |
|---|---|---|---|---|
| BF16 | 40.7 | 122.7 (γ=8) | **3.01x** | 33% |
| BF16 | 40.7 | 118.3 (γ=4) | 2.91x | 56% |
| fp8 (런타임) | 68.4 | 200.3 (γ=4) | **2.93x** | 61% |
| qat (w4a16) | 89.8 | 235.0 (γ=4, 전용 드래프터) | **2.62x** | 56% |

- 가장 빠른 조합: **qat + MTP = 235 tok/s** (baseline BF16 40.7의 5.8배).
- baseline 40.7 = JarvisLabs 40.3, MTP γ=8 3.01x = JarvisLabs 3.11x — 제3자 벤치(JarvisLabs)와 일치.
  단, 양쪽 다 동일 vLLM 스택이라 완전 독립 검증은 아님(공통 편향 가능). 측정 신뢰의 1차 근거는
  vLLM 내부 `Mean acceptance length` 지표 + non-streaming 재측정의 논리적 정합성.
- diffusion(H100, `:gemma` 이미지, `--max-num-seqs 4`): 8K prefill 701ms / 128K prefill 2,679ms,
  순수 generation decode ~836 tok/s(짧은입력) / 311 tok/s(8K 컨텍스트). A100에서 크래시하던
  동시성(conc=4 정상)·128K(정상) 전부 H100에선 작동.

### 측정 교훈 (3건, 모두 H100 단계에서 발견)
1. **streaming은 spec decode를 과소측정** → non-streaming(forced 출력 / wall-clock, prefill 분리)으로 측정해야 함.
2. **유령 systemd 서비스**: A100 스냅샷에 `ops-llm-test.service`가 따라와 H100 부팅 시 자동 기동 →
   호스트 vLLM이 GPU 33GB 점유 → fp16 "메모리 부족" 실패의 진짜 원인. `systemctl disable`로 해결.
   (31B BF16은 H100에서 잘 돈다 — 모델 크기 문제가 아니었음.)
3. **cudagraph 메모리 프로파일링**(vLLM v0.21+ 기본): 61GB 가중치 + cudagraph 예약이 KV를 줄여
   util 0.90에선 16384 컨텍스트 불가 → util 0.95 또는 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.

### A100 baseline (유효 — non-MTP라 streaming≈non-streaming)
fp16 22.8 / fp8 37.5 / qat 52.8 tok/s. H100 대비 비율: fp8 1.78x, qat 1.65x, fp16 1.76x.
(A100의 MTP·diffusion-conc·128K 항목은 무효 또는 크래시.)

---

## H100 종합 매트릭스 (정본 v2, 2026-06-13 — 전 모델 × 정밀도 × 기법 + 정확도)

**측정:** 단일 H100 80GB (Spot, us-central1-a), vLLM `nightly`(AR/MTP) · `:gemma`(diffusion),
non-streaming 총처리량(고정 출력 512토큰 / wall-clock latency, `--no-enable-prefix-caching`).
MTP = γ4 (`num_speculative_tokens=4`).
**정확도:** ops 근본원인 식별 태스크 — 한 서비스가 먼저 죽고 나머지가 캐스케이드되는 합성 인시던트
로그에서 근본원인 서비스를 보기 중 선택, ground-truth 대조, N=30.

### 속도 (tok/s, short context [8K context])

| 모델 (활성 param) | 정밀도 | AR base | AR + MTP (가속) | diffusion |
|---|---|---|---|---|
| **26B-A4B** (MoE ~4B) | BF16 | 199.8 [187.7] | 322.9 [312.9] (1.62x) | 615.5 |
| | fp8 | 226.0 [211.6] | 390.3 [337.9] (1.73x) | **864.4** |
| | int8 | 218.2 [200.9] | 381.6 [331.2] (1.75x) | 618.4 [573.3] |
| **31B** (dense) | BF16 | 40.5 [37.8] | OOM* | — |
| | fp8 | 67.4 [61.8] | 178.1 [150.2] (**2.64x**) | — |
| | qat w4a16 | 89.6 [70.7] | 213.7 [143.1] (2.38x) | — |
| **12B** (dense) | BF16 | 82.4 [78.1] | 186.5 [157.9] (2.26x) | — |
| | fp8 | 118.7 [111.6] | 200.7 [178.6] (1.69x) | — |
| **Qwen3.6 35B-A3B** (MoE ~3B) | fp8 | 216.7 [204.0] | 315.6 (γ2) / 240.4 (γ1) | n/a |
| **Qwen3.6 27B** (dense) | fp8 | 14.3** | DIED** | n/a |

*\*31B-bf16-MTP: cudagraph 켜면 KV 부족 OOM(62GB 가중치+드래프터 동시적재 시 KV<6.89GB 필요),
`--enforce-eager`로만 기동되나 그 수치(21.4/58.9 tok/s)는 커널런치 오버헤드로 다른 MTP 수치와 비교
불가 → **단일 H100 80GB에서 깨끗한 측정 불가**로 결론. fp8/qat에선 MTP 정상(2.64x/2.38x)이라 실무상
무의미한 코너(31B는 프로덕션에서 bf16 대신 fp8/qat 사용).*
*\*\*Qwen 27B-FP8(`Qwen3_5MTP` arch): vLLM nightly 비호환 — base는 `--enforce-eager`로 14 tok/s만,
MTP는 DIED. (원래 0.22.1에선 작동.) nightly 스택 한계로 미측정 문서화.*

### 정확도 (N=30, ops 근본원인 태스크)

| 모델 | 정밀도 | AR/base 정확도 | diffusion 정확도 |
|---|---|---|---|
| 26B-A4B | BF16 / fp8 / int8 | 100% / 100% / 100% | 86.7% / 90.0% / **93.3%** |
| 31B | bf16 / fp8 / qat | 100% / 100% / 100% | — |
| 12B | bf16 / fp8 | 100% / 100% | — |
| Qwen35-A3B | fp8 | 6.7%* | — |

*\*Qwen 6.7%는 실제 성능이 아니라 **채점 파서 아티팩트**: Qwen이 답 앞에 추론 서두("the user wants to
identify the...")를 붙이는데 추출기가 서비스명 대신 그 서두를 긁음. Gemma 계열은 답 포맷을 지켜 100%.
Qwen-aware 추출기로 재측정 필요(미수정).*

### 핵심 결론
1. **MTP는 전 dense 모델·정밀도에서 일관된 가속, 정확도 무손실**(AR 100% 불변). dense일수록 이득 큼:
   31B fp8 2.64x > 12B bf16 2.26x > 26B-MoE 1.6~1.7x. dense는 매 토큰 전 파라미터를 로드해 대역폭
   병목이 크고, spec decode가 그 병목을 평균 수락길이만큼 분할상환하기 때문.
2. **양자화는 정확도를 떨어뜨리지 않음**: fp8/int8/qat 전부 base와 동일(100%). 속도만 단조 상승
   (31B base: bf16 40.5 → fp8 67.4 → qat 89.6).
3. **diffusion은 속도 최강(26B-fp8 864 tok/s)이나 정확도는 86.7~93.3%로 유일하게 100% 미달**.
   int8이 diffusion 중 정확도 최고(93.3%)이나 속도(618)는 fp8(864)보다 낮음. **diffusion int8 지원 확인.**
4. **속도-정확도 동시 최적 = 26B-A4B-fp8**: AR 226(100%) / MTP 390(100%) / diffusion 864(90%).
   MoE의 적은 활성 파라미터(~4B)가 처리량·feasibility 양쪽에 유리.

---

> [!NOTE]
> 아래 §1 인프라 여정과 §10 부록은 유효. §3~5의 A100 MTP "손해"/acceptance 24-57% 서술은
> streaming 아티팩트로 **폐기** — 위 H100 정본 표로 대체.

---

## 1. 인프라 여정 (재현에 필수)

Gemma 4(2026-06 출시)는 tooling이 따라잡는 중이라, 작동 환경을 찾는 것 자체가 실험의 절반이었다.

### 1.1 막다른 길들

| 시도 | 결과 |
|---|---|
| vLLM 0.22.1 (pip 최신 stable) | `gemma4_unified`를 native 미인식 → `TransformersMultiModalForCausalLM` 폴백 → torch.compile(dynamo) 크래시 |
| transformers 업그레이드 (5.11.0) | 폴백 여전 — vLLM에 native 클래스가 없는 게 근본 원인 |
| docker `vllm/vllm-openai:gemma4-0505-cu129` (5월 이미지) | transformers가 구버전 → 6월 `gemma4_unified` 체크포인트의 vision config 미인식 |

근본 원인: **체크포인트 포맷이 5월(`gemma4`) → 6월(`gemma4_unified`)로 바뀌었고**, native 지원은 그 이후 빌드에만 존재.

### 1.2 작동한 레시피 (확정)

| 기법 | 이미지 | 결정적 로그 |
|---|---|---|
| original / QAT / MTP | **`vllm/vllm-openai:nightly`** | `Resolved architecture: Gemma4UnifiedForConditionalGeneration` + `Gemma4MTPModel` |
| diffusion | **`vllm/vllm-openai:gemma`** (전용) | `Resolved architecture: DiffusionGemmaForBlockDiffusion` |

```bash
# MTP (드래프터 speculative decoding)
docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 vllm/vllm-openai:nightly \
  --model google/gemma-4-31B-it --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'

# diffusion (전용 이미지)
docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 vllm/vllm-openai:gemma \
  --model google/diffusiongemma-26B-A4B-it --max-model-len 131072 --max-num-seqs 8
```

### 1.3 Gemma 4 MTP의 메커니즘 (Qwen과 다름)

- Qwen3.6: 모델에 **내장된 MTP 헤드**.
- Gemma 4: 타깃마다 짝이 되는 **별도 ~0.4B dense 드래프터**(`gemma-4-<size>-it-assistant`, vocab 262K 공유). draft-model speculative decoding.
- Google이 원래 이 MTP를 공개 배포판에서 제거했다가, 커뮤니티 리버스 엔지니어링(`.litertlm` 분석) 후 2026-05-05 드래프터를 별도 릴리스.

---

## 2. 측정 방법론

- **2개 컨텍스트**: 8K(decode 의미있는 구간) / 128K(~123K 토큰, prefill 지배)
- **변종별 max 동시성**: vLLM이 보고하는 KV 캐시 총량으로 각 컨텍스트의 saturation 동시성을 자동 계산
- temp=0, max_tokens=1024
- single은 streaming(TTFT 측정), 동시성은 non-streaming(서버 throughput 정확)
- 프롬프트는 문자 예산 기반 생성 후 `/tokenize`로 실제 토큰 수 검증 (로그 텍스트 1.83자/토큰)
- 각 페이즈(변종) 사이 컨테이너 완전 정지 + GPU 0MB 해제 검증

---

## 3. 결과 — 26B-A4B (MoE, 3.8B active)

| 기법 | 8K single decode | 8K conc agg | 128K single decode | 128K TTFT | 128K conc agg | acceptance |
|---|---|---|---|---|---|---|
| **original** (BF16) | 110.4 tok/s | 670.5 (c=118) | 64.3 tok/s | 44.8s | 19.4 (c=8) | — |
| **MTP** (drafter n=4) | 45.1 tok/s | 789.7 (c=107) | 10.5 tok/s | 52.8s | 16.4 (c=7) | **52.3%** |
| **diffusion** | **192.7 tok/s** | 크래시 | 크래시(HTTP500) | — | 크래시 | — |
| 128K max 동시성 | 6.53x | 5.7x(KV↓) | — | — | — | — |

핵심:
- **diffusion이 8K single 최속(192.7, original의 1.75배)** — diffusion 속도 주장 실측 확인. 단 **A100에서 동시성/128K는 엔진 크래시**(`EngineDeadError`).
- MTP single decode는 손해(0.41x@8K, 0.16x@128K). acceptance 52%로 낮고 출력 길이 비대칭(original 915 vs MTP 297 토큰 @temp=0 → 비무손실).

---

## 4. 결과 — 31B (Dense)

| 기법 | 8K single decode | 128K | 128K max 동시성 | acceptance | feasibility |
|---|---|---|---|---|---|
| **original** (BF16) | 22.8 tok/s | **불가 (OOM)** | — | — | 128K 1개도 못 띄움 |
| **MTP** (BF16+drafter) | 17.1 tok/s | **불가 (OOM)** | — | 56.9% | 128K 불가 |
| **QAT** (w4a16) | **52.8 tok/s** | 30.8 tok/s | 3.88x | — | **유일하게 128K 가능** |
| QAT 128K TTFT | — | **193초(!)** | — | — | dense라 prefill 극도로 비쌈 |

핵심:
- **31B BF16은 단일 A100 80GB에서 128K를 못 돌린다.** vLLM 에러: *"max seq len 131072엔 12.7 GiB KV 필요하나 10.26 GiB만 가용. 추정 최대 컨텍스트 99,136."* MTP는 드래프터까지 얹어 더 불가.
- **QAT(w4a16)만이 128K 서빙 가능** (동시 3.88개).
- **QAT 8K single(52.8) > BF16(22.8) — 2.3배 빠름.** decode가 메모리대역폭 바운드라 4-bit 가중치가 4배 적게 로드됨. QAT는 메모리뿐 아니라 dense decode 속도의 압도적 레버.

[핵심 발견] dense에서 QAT는 "삼관왕"
  1. 속도: 8K single 2.3배 (52.8 vs 22.8)
  2. 메모리: 가중치 61GB -> 20GB
  3. Feasibility: BF16은 128K 불가, QAT만 가능

---

## 5. 4기법 종합 비교

### 5.1 단문맥(8K) single-stream decode 속도

| 기법 | 26B-A4B | 31B |
|---|---|---|
| original | 110.4 | 22.8 |
| MTP | 45.1 (0.41x) | 17.1 (0.75x) |
| QAT | (해당없음) | 52.8 (2.3x) |
| diffusion | 192.7 (1.75x) | (해당없음) |

### 5.2 장문맥(128K) feasibility

| 기법 | 26B-A4B | 31B |
|---|---|---|
| original | 동작 (TTFT 44.8s, 동시 6.5) | **불가 (OOM)** |
| MTP | 동작 (이득 없음) | **불가** |
| QAT | (해당없음) | 동작 (TTFT 193s, 동시 3.88) |
| diffusion | **크래시** | (해당없음) |

---

## 6. 메커니즘 분석

### 6.1 왜 MTP가 또 손해인가 (Qwen에 이어)

- **acceptance가 낮다(52-57%)**: Gemma의 작은 별도 0.4B 드래프터는 Qwen의 내장 MTP 헤드(70-86%)보다 예측 정확도가 떨어진다. num_speculative_tokens=4로 멀리 던질수록 거부 낭비 큼.
- **128K에선 prefill이 압도**: single TTFT 44-193초 vs decode 수 초. MTP는 decode만 가속 -> 효과 ~ 0.
- **비무손실**: temp=0인데 MTP 출력 길이가 original과 다름(915 vs 297). FP/draft 경로 수치 차이로 발산.

### 6.2 왜 diffusion이 빠른데 못 쓰나

- block diffusion이 256-token canvas를 병렬 복원 -> 단문맥 single에서 192 tok/s(최속).
- 하지만 A100(FP8 텐서코어 없음, 공식 미검증)에서 **동시성/장문맥 시 엔진 코어 크래시**. H100/H200 전용 최적화의 한계.

### 6.3 왜 QAT가 dense의 정답인가

- dense는 토큰당 전체 가중치를 로드 -> decode가 메모리대역폭 바운드. 4-bit는 4배 적게 로드 -> 2.3배 빠름.
- 가중치 61GB->20GB로 KV 여유 확보 -> 단일 A100에서 128K 서빙 가능해짐.
- (MoE 26B-A4B는 expert_dim=704이 작아 공식 w4a16 미제공 -> 이 레버를 못 씀.)

---

## 7. Qwen MTP 실험과의 대조

| | Qwen3.6 MTP | Gemma 4 MTP |
|---|---|---|
| 방식 | 내장 MTP 헤드 | 별도 0.4B 드래프터 |
| acceptance | 65-86% | 52-57% (더 낮음) |
| single decode | 손해 (0.44-0.65x) | 손해 (0.41-0.75x) |
| 결론 | MTP off가 정답 | MTP off가 정답 |

**두 패러다임 모두, 우리 워크로드(대용량 입력 + 단일 A100)에서 MTP는 순손실.** 메커니즘은 다르지만(Qwen=MoE expert union 비용 / Gemma=낮은 acceptance + prefill 지배) 결론은 동일.

---

## 8. 실무 권고

1. **장문맥(128K급) dense 서빙 = QAT 필수**, 선택 아님. 31B BF16은 단일 A100에서 128K 불가.
2. **MTP는 켜지 말 것** (양 모델 패밀리, 양 컨텍스트에서 손해). Gemma 드래프터 acceptance가 특히 낮음.
3. **diffusion은 단일 사용자·단문맥·저지연 용도**로만. A100 production 동시성엔 부적합(엔진 크래시) — H100+ 필요.
4. **MoE(26B-A4B) vs Dense(31B)**: MoE가 decode 훨씬 빠름(110 vs 23 @8K). 단 MoE는 QAT w4a16 미지원으로 장문맥 메모리 레버가 없음.

---

## 9. 한계

- 단일 A100 80GB, 단일 워크로드(ops 분석). FP8 가속 없는 Ampere라 H100과 절대 수치 다름.
- diffusion은 A100 비공식 -> 크래시가 모델 결함이 아니라 HW/이미지 미성숙 탓일 수 있음.
- MTP num_speculative_tokens=4 고정 — 더 낮은 값(1-2)에서 결과 다를 수 있으나, 우리 측정 + acceptance 추세상 이득 전환 가능성 낮음.
- 8K 동시성 aggregate throughput은 큐잉(queue wait)을 포함해 "시스템 처리량"이지 순수 decode가 아님. single decode tok/s가 가장 깨끗한 지표.
- 26B-A4B QAT(w4a16)는 공식 미제공이라 MoE의 QAT 축은 측정 불가(GGUF는 런타임이 달라 confound).

---

## 10. 부록

### 10.1 디렉토리

```
test_mtp/
├── GEMMA4.md              # 본 보고서
├── gemma4_logs/
│   ├── benchmark_gemma.py # 2컨텍스트 x 변종별 max 동시성 벤치
│   ├── gbench-26B-orig.log / gbench-26B-mtp.log / gbench-26B-diffusion.log
│   ├── gbench-31B-qat.log / gbench-31B-orig.log (128K OOM)
│   └── gbench-31B-orig-16k.log / gbench-31B-mtp-16k.log (8K 데이터)
└── README.md              # Qwen MTP 실험 (선행)
```

### 10.2 모델별 포맷 가용성 (조사 결과)

| 모델 | original(BF16) | QAT w4a16-ct(safetensors) | QAT q4_0(GGUF) | diffusion |
|---|---|---|---|---|
| 12B (dense) | O | O | O | — |
| 31B (dense) | O | O | O | — |
| 26B-A4B (MoE) | O | **없음**(expert 작음) | O | O (diffusiongemma) |

### 10.3 환경 핵심값

- torch 2.11.0+cu130, vLLM nightly(0.22.1rc1.dev357 계열) / `:gemma` 이미지
- nvidia-container-toolkit 1.17.8 (DL VM 기본), docker 29.1.3
- transformers 5.11.0
