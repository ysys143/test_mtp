# MTP (Multi-Token Prediction) 실측 실험 보고서

**대상:** Qwen3.6-27B-FP8 (Dense) / Qwen3.6-35B-A3B-FP8 (MoE)
**환경:** 단일 A100 80GB, vLLM, GCP
**측정일:** 2026-06-11
**워크로드:** ops 분석(SRE 인시던트 근본원인 분석) 모사 — 대용량 입력(~3,900 토큰) + 구조화 출력

---

> [!WARNING]
> **이 보고서의 "MTP 손해" 결론은 측정 방법 결함으로 무효일 가능성이 매우 높다 (2026-06-13 발견).**
> 모든 decode 수치를 **streaming `decode_tok/s`**(첫토큰~마지막토큰 토큰율)로 측정했는데, 이 방식은
> speculative decoding이 토큰을 **버스트로 커밋**하는 것을 약 3배 과소측정한다. 후속 Gemma 4 실험에서
> 동일 모델/설정을 **non-streaming**(고정 출력 토큰수 / wall-clock, prefill 분리)으로 재측정하니 MTP가
> **0.8x 손해 → 2.6~3.0x 이득**으로 뒤집혔다 (`GEMMA4.md` §H100 참조). Qwen3.6 MTP도 같은 streaming
> 방법으로 쟀으므로 "0.44x 손해" 결론은 **재측정 전까지 신뢰 불가**. acceptance(65~86%)는 정상이었고,
> 낮은 throughput은 측정 아티팩트였을 개연성이 크다. **재측정 완료(2026-06-13, §0.5) — MTP는 이득으로 정정됨.**

## 0. 한 줄 결론 (※ 위 경고로 무효화됨 — 재측정 대기)

> ~~**우리 운영 워크로드(대용량 입력 + 동시성 + FP8 + 단일 A100)에서는, 모델 아키텍처(Dense/MoE)와 gamma 값에 관계없이 MTP가 항상 순손실(net-negative)이다.**~~ (streaming 측정 아티팩트로 무효)

~~acceptance rate는 65~86%로 건강했음에도 throughput은 매번 떨어졌다.~~ → acceptance는 정상이었고, throughput "손해"는 streaming 측정 결함이었을 가능성이 큼. non-streaming 재측정 필요.

---

## 0.5 H100 non-streaming 재측정 — 정정본 (2026-06-13)

위 경고대로 H100에서 **non-streaming**(고정 출력 512토큰 / wall-clock latency, `--no-enable-prefix-caching`)으로
Qwen3.6 MTP를 재측정했다. **결과: "MTP 손해" 결론은 뒤집혔다 — MTP는 이득이다.**

| 모델 | base tok/s (short [8K]) | + MTP tok/s | 가속 |
|---|---|---|---|
| Qwen3.6 35B-A3B-FP8 (MoE ~3B) | 216.7 [204.0] | γ1 240.4 / γ2 315.6 | 1.11x / **1.46x** |
| Qwen3.6 27B-FP8 (dense) | 14.3 (eager only)* | DIED* | 측정불가 |

*\*27B-FP8(`Qwen3_5MTP` arch)은 vLLM nightly와 비호환: base는 `--enforce-eager`로 14 tok/s만 나오고
MTP는 엔진 초기화 실패(DIED). 원래 A100 0.22.1 스택에선 작동했으므로 nightly 회귀로 추정. nightly는
Gemma 4(`gemma4_unified`) 네이티브 지원 때문에 불가피하게 선택 — Qwen 27B는 이 스택에서 미측정 문서화.*

**해석:** 같은 streaming→non-streaming 정정이 Qwen MoE에도 동일하게 적용된다. 단 Qwen35-A3B의 MTP
이득(1.46x)은 Gemma dense(2.6x+)보다 작은데, MoE(활성 ~3B)는 이미 대역폭 부담이 작아 spec decode의
분할상환 여지가 적기 때문 — `GEMMA4.md` §H100 종합 매트릭스의 MoE(1.6~1.7x) vs dense(2.6x) 패턴과 일치.
(전 모델×정밀도×기법 + 정확도 종합 매트릭스는 `GEMMA4.md` 정본 v2 참조.)

---

## 1. 배경 및 목적

### 1.1 동기

Qwen3.6 계열이 MTP(Multi-Token Prediction)를 지원하는 것이 확인되어(별도 조사), 실제 운영 추론 노드와 동일한 하드웨어/런타임에서 **MTP가 우리 워크로드에 실질적 이득을 주는지** 직접 측정하기로 했다.

### 1.2 MTP / speculative decoding 개념

MTP는 메인 모델에 부착된 보조 헤드(draft head)가 앞으로의 토큰 여러 개를 미리 추측(draft)하고, 메인 모델의 forward 1회로 그 추측들을 동시에 검증(verify)하는 speculative decoding 기법이다.

- `num_speculative_tokens` (= **gamma**): 한 사이클에 미리 추측하는 토큰 수
- 메인 모델 forward 1회로 최대 `gamma+1`개 토큰을 확정(전부 채택 시)
- 이론적 사이클당 기대 토큰 수 (acceptance율 α): `1 + α + α² + ... + α^gamma`

### 1.3 사전 가설 (조사 기반, 이후 실측으로 검증/반증됨)

| 가설 | 출처 | 실측 결과 |
|---|---|---|
| MoE는 MTP acceptance가 낮아(~11%) 손해 | zolotukhin.ai 추정치 | **반증** — 실측 83% (높음). 손해 원인은 acceptance가 아니라 검증 비용 |
| Dense는 MTP로 2.24~2.40x 이득 (gamma=2~3) | 커뮤니티 벤치마크 | **재현 실패** — 우리 셋업에선 전 gamma 손해 |
| MoE는 MTP에 불리, Dense는 유리 | 일반론 | **부분 확인** — Dense 페널티가 MoE보다 "덜 나쁨"(아키텍처 효과 실재). 단 둘 다 1.0x 미달 |

---

## 2. 실험 환경 (인프라)

### 2.1 테스트 VM

기존 운영 인프라를 **전혀 건드리지 않고** 별도 테스트 VM(`llm-test`)을 신규 생성했다.

| 항목 | 값 |
|---|---|
| 인스턴스 | `llm-test` |
| 머신 타입 | `a2-ultragpu-1g` |
| GPU | NVIDIA A100 80GB × 1 |
| Zone | `asia-southeast1-c` |
| 내부 IP | `10.20.0.10` |
| 외부 IP | `35.198.209.150` |
| 이미지 | `deeplearning-platform-release/pytorch-2-9-cu129-ubuntu-2204-nvidia-580` |
| 디스크 | 200GB pd-ssd |
| 런타임 | vLLM (`>=0.9.0`), v1 엔진 기본 플래그 |

> **주의 (FP8 + A100):** A100은 Ampere 아키텍처로 **네이티브 FP8 텐서코어가 없다**(FP8 네이티브는 Hopper/H100부터). vLLM의 FP8은 weight-only 양자화로, 가중치는 FP8로 저장(메모리·대역폭 절감)하되 행렬연산은 BF16/FP16으로 디퀀타이즈해 수행한다. 즉 FP8은 메모리/대역폭은 줄여도 연산 가속은 없다 — MTP 경제성에 영향을 주는 요인.

### 2.2 인프라 변경 사항 (terraform)

기존 인프라 무변경 원칙으로, VM 하나만 추가:

- `dev-env/terraform/main.tf`: `google_compute_instance.llm_test` 리소스 추가 (기존 `llm` 블록 패턴 복제, IP만 `.10`으로 변경, 기존 `ops-llm` 방화벽 태그 재사용 → 신규 방화벽 룰 불필요)
- `dev-env/cloud-init/llm-test.sh`: 모델/MTP를 env로 전환 가능한 래퍼 기동 스크립트

적용:
```bash
terraform apply -target=google_compute_instance.llm_test
```
`Plan: 1 to add, 0 to change, 0 to destroy` 확인 후 적용 (기존 VM 무변경 검증).

### 2.3 쿼터 이슈

최초 apply 시 실패:
```
Error: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 1.0 in region asia-southeast1.
```
기존 운영 `llm` VM이 유일한 A100 80GB 쿼터(1)를 점유 중이었다. 쿼터를 **2로 증설** 후 재적용하여 생성 성공.

---

## 3. 측정 방법론

### 3.1 워크로드 (현실 모사 프롬프트)

운영 엔진의 실제 사용 패턴(로그/메트릭/트레이스를 주입해 근본원인 분석 리포트 생성)을 모사하도록, 다음을 프로그램적으로 생성한 **대용량 프롬프트**를 사용:

- 160개 로그 라인 (INFO/WARN/ERROR 혼합, 타임스탬프·trace_id·latency 포함)
- 9개 서비스의 메트릭 baseline 편차
- 근본원인/blast-radius/완화책/심각도 분석 요청
- **입력 약 3,900 토큰** (prefill-heavy 워크로드)

### 3.2 측정 지표

- **TTFT** (Time To First Token): 요청 전송 → 첫 출력 토큰 수신
- **TTLP** (Time To Last Token / 전체 완료 시간)
- **tok/s** (단일 요청): 생성 구간 토큰/초
- **agg tok/s** (동시성): 전체 출력 토큰 합 / wall-clock
- **acceptance rate**: vLLM `/metrics`의 `spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total` 직접 추출

### 3.3 통제 조건

- `temperature=0` (greedy) — v2 이후. 샘플링 변동 제거
- `enable_thinking=False` — reasoning 토큰 폭주 방지, 순수 생성 측정
- `max_tokens=512`
- 동시성 스윕: single / conc=8 / conc=16 (non-stream은 별도 미사용; streaming SSE)
- 워밍업 1회 후 측정 (torch.compile/캐시 안정화)

### 3.4 메모리 정리 프로토콜 (중요)

각 페이즈(MTP on/off, gamma 변경) 사이에 **systemd 서비스 완전 정지 → GPU 메모리 0MB 도달 검증** 후 다음 페이즈 시작. 모든 페이즈 전환에서 `GPU freed (0MB)` 확인됨 → 페이즈 간 상태 오염 없음 보장.

---

## 4. 실험 연대기 및 결과

### 4.1 초기 셋업 및 OOM (학습)

- 최초 `Qwen3.6-27B` **BF16** 시도 → **CUDA OOM**
  - 가중치 51.89 GiB + `max_model_len=131072` KV 캐시 > 80GB
- → **FP8 전환**(`Qwen3.6-27B-FP8`, 가중치 ~26GB)으로 해결. 이후 모든 실험 FP8.
- vLLM 로그에서 **MTP 헤드 정상 감지** 확인:
  ```
  Detected MTP model. Sharing target model embedding weights with the draft model.
  Detected MTP model. Sharing target model lm_head weights with the draft model.
  ```

### 4.2 실험 A — A3B-FP8, v1 (샘플링, gamma=1)

MoE 모델 순차 비교 (sampling temp 기본, max_tokens 가변). **첫 신호**: MTP가 오히려 느림.

| 지표 | NO-MTP | MTP (gamma=1) |
|---|---|---|
| TTFT | 534 ms | 567 ms |
| tok/s | 155.7 | 65.9 |

→ temp 미고정·출력 길이 비대칭 한계로, v2에서 엄밀 재측정.

### 4.3 실험 B — A3B-FP8, v2 (temp=0 + 동시성 + acceptance)

`Qwen/Qwen3.6-35B-A3B-FP8`, MoE (35B 총 / 3B 활성, 256 expert 중 8 활성).

| 시나리오 | NO-MTP | MTP (gamma=1) | 비율 |
|---|---|---|---|
| single tok/s | 160.9 | 70.0 | **0.44x** |
| conc=8 agg tok/s | 446.3 | 160.0 | 0.36x |
| conc=16 agg tok/s | 554.7 | 259.3 | 0.47x |
| single TTFT | 534 ms | 556 ms | ~동일 |
| **acceptance** | — | **83.1%** | — |

**핵심:** acceptance 83%인데도 전 동시성에서 2~2.8배 느림. 사전 조사의 "MoE ~11% acceptance" 주장 반증.

### 4.4 실험 C — Dense 27B-FP8, v3 (temp=0 + 동시성, gamma=1)

`Qwen/Qwen3.6-27B-FP8`, Dense. **A3B와 완전히 동일한 조건** → 아키텍처만 변수로 격리.

| 시나리오 | NO-MTP | MTP (gamma=1) | 비율 |
|---|---|---|---|
| single tok/s | 48.2 | 31.3 | **0.65x** |
| conc=8 agg tok/s | 125.5 | 64.4 | 0.51x |
| conc=16 agg tok/s | 147.9 | 77.0 | 0.52x |
| single TTFT | 2,365 ms | 2,406 ms | ~동일 |
| **acceptance** | — | **86.0%** | — |

**핵심:** Dense도 MTP가 손해. 단 페널티(0.65x)가 MoE(0.44x)보다 **덜 나쁨** → 아키텍처 효과 실재.

### 4.5 실험 D — Dense 27B-FP8, v4 (gamma 스윕 2/3)

literature의 이득 사례가 gamma=2~3이라 추가 측정.

| 설정 | single tok/s | vs baseline | conc=8 | conc=16 | acceptance |
|---|---|---|---|---|---|
| NO-MTP | **48.2** | 1.00x | 125.5 | 147.9 | — |
| gamma=1 | 31.3 | 0.65x | 64.4 | 77.0 | 86.0% |
| gamma=2 | 31.3 | 0.65x | 53.7 | 60.8 | 75.3% |
| gamma=3 | 30.3 | 0.63x | 47.4 | 51.9 | 65.3% |

**핵심:** gamma를 올릴수록 acceptance가 하락(86→75→65%)하고 throughput도 같이 악화. literature의 "dense gamma=2에서 2.24x"는 우리 박스에서 재현되지 않음.

---

## 5. 종합 결과 (같은 A100, single-stream 기준)

| 모델 | 구조 | baseline | MTP 최선 | 비율 | MTP acceptance |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-FP8 | MoE (3B active) | 160.9 | 70.0 (g1) | 0.44x | 83.1% |
| Qwen3.6-27B-FP8 | Dense | 48.2 | 31.3 (g1·g2) | 0.65x | 86.0% |

> baseline 절대속도: A3B(160.9) > Dense 27B(48.2). 정상이다 — A3B는 토큰당 3B만 활성화하므로 27B 전체를 쓰는 dense보다 훨씬 빠르다.

**모든 조합(2 모델 × {gamma 1,2,3} × {동시성 1,8,16})에서 MTP는 baseline 미달.**

---

## 6. 메커니즘 분석 — 왜 MTP가 손해인가

### 6.1 acceptance ≠ throughput (MoE 전문가 union 비용)

MoE(A3B)는 토큰당 256 expert 중 8개만 활성화한다. MTP가 draft한 K개 토큰을 검증할 때, 각 토큰이 **서로 다른 expert 집합**을 건드리면 그 **합집합(union)을 모두 로드**해야 한다. 전문가 포화 임계값은 약 94 토큰으로, 우리 batch(1~16)는 그 아래라 매 검증이 거의 풀모델 대역폭을 소모한다. → acceptance가 높아도(83%) 검증 비용이 이득을 잡아먹음. (MoESD 논문 arxiv 2505.19645, Cascade 논문 2506.20675가 정식 증명)

### 6.2 gamma 증가의 역효과

gamma↑ → 뒤쪽 draft 토큰의 acceptance 하락(α^gamma) → 거부된 draft의 검증 연산은 그대로 낭비 + 검증 batch 확대. 우리 실측에서 gamma 1→3으로 acceptance 86→65% 하락, throughput도 동반 하락.

### 6.3 prefill 지배 워크로드

우리 입력은 ~3,900 토큰. 단일 요청 TTFT가 2.4초(생성은 ~10초). MTP는 **decode만** 가속하고 **prefill은 전혀 못 줄인다**. 입력이 큰 워크로드에선 MTP 이득 여지가 작고, 오버헤드만 추가된다.

### 6.4 셋업 요인 (literature와의 차이)

| 변수 | literature 이득 사례 | 우리 측정 |
|---|---|---|
| 입력 길이 | 짧음 (decode 지배) | ~3,900 토큰 (prefill 지배) |
| chunked prefill | `--no-enable-chunked-prefill` (권장) | vLLM v1 기본 ON (미설정) |
| 정밀도 | 주로 BF16 | FP8 (A100는 FP8 연산 가속 없음) |
| 하드웨어 | 다양 (저대역폭에서 MTP 유리) | A100 80GB HBM2e ~2TB/s |

### 6.5 FP8 MTP 경로 non-lossless

`temperature=0`(greedy)인데도 NO-MTP와 MTP의 **출력 길이가 다름**(예: 512 cap vs 213~285 자연종료). MTP가 무손실(bit-exact)이어야 하나, FP8 양자화 경로에서 draft+verify 커널의 수치 차이로 한 토큰이 갈리면 이후가 발산한다. → A100/FP8에서 vLLM MTP는 완전 무손실이 아니며 출력 분포에도 미세 영향 가능. (vLLM 이슈 #36872와 일치)

---

## 7. 외부 증거 (literature 대조)

우리 실측이 단발 이상치가 아님을 뒷받침하는 공개 사례:

| 출처 | 환경 | 결과 |
|---|---|---|
| vLLM #35387 | Qwen3-Next-80B-A3B-FP8, 4×H100 | MTP-2 시 **+76% latency** (같은 A3B 계열) |
| vLLM #36872 | Qwen3.5-35B-A3B-FP8 | FP8+MTP **출력 깨짐 + acceptance 붕괴** |
| vLLM #21278 | Qwen3-32B-FP8 + draft | spec decode **-50% throughput** |
| SGLang #21138 | Nemotron MoE | NEXTN **-24%**, 2 draft 토큰 전부 거부 |
| MoESD 논문 (2505.19645) | Qwen2-57B-A14B MoE | batch<16에서 **baseline 미만** (expert union) |
| Cascade 논문 (2506.20675) | MoE 일반 | 검증 2~3배 비용, **최대 1.5x 슬로다운** |
| thc1006 (RTX3090) | Qwen3.6-35B-A3B | **100% acceptance인데도 -3~39%** |
| 반례: DGX Spark GB10 | Qwen3.6-35B-A3B-FP8 | 273GB/s 저대역폭에선 MTP-1 **+22~24%** (HW 차이) |

vLLM에는 고동시성에서 spec decode를 자동 비활성화하는 `--speculative-disable-by-batch-size`가 존재 — "고동시성에서 spec decode가 표준 디코딩보다 느려진다"는 것을 공식 인정하는 기능.

---

## 8. 결론 및 권고

### 8.1 결론

1. **우리 워크로드에서 MTP는 Dense/MoE, gamma 무관하게 순손실.**
2. acceptance(65~86%)는 건강했으나 throughput은 매번 하락 — 검증 비용이 핵심 변수.
3. 아키텍처 효과는 실재 — Dense 페널티(0.65x)가 MoE(0.44x)보다 덜 나쁨.
4. 손해 원인 스택: prefill 지배 + MoE expert union + chunked-prefill 기본 ON + FP8(A100 연산 가속 없음) + gamma 증가의 역효과.

### 8.2 운영 권고

| 권고 | 신뢰도 | 내용 |
|---|---|---|
| **MTP off 유지** | HIGH | 운영 `llm` VM 현 구성(MTP 미사용)이 정답. `--speculative-config` 미사용 |
| **prefix caching로 방향 전환** | HIGH | `--enable-prefix-caching` — 공유 시스템 프롬프트 재사용으로 TTFT 직접 단축. prefill-heavy에 맞는 레버 (단 MTP와 상호배타) |
| 굳이 MTP 평가 시 | MED | `--no-enable-chunked-prefill` 필수, `disable_by_batch_size`로 고부하 자동 차단, **Dense + 짧은입력/긴출력 + 저동시성**에서만 |
| ngram 대안 | LOW | `method:ngram` — 위험 없는 소폭 이득 옵션 |

### 8.3 MTP가 유리한 조건 (참고)

| 조건 | MTP 유리? |
|---|---|
| 단일 요청 / 저동시성 + 짧은 입력 + 긴 출력 (decode 지배) | O |
| 동시성 / prefill 지배 (대용량 입력) | X (우리 케이스) |
| Dense 모델 | 상대적 유리 (그래도 우리 셋업선 미달) |
| MoE 모델 | 불리 (expert union 비용) |
| 저대역폭 HW (GB10 등) | O |
| 고대역폭 HBM (A100/H100) | 상대적 불리 |

---

## 9. 한계

- **단일 박스, 단일 워크로드** — 결론은 "우리 ops 분석 워크로드 + 단일 A100 + FP8 + vLLM v1 기본 플래그"에 한정. 일반화 시 주의.
- `--no-enable-chunked-prefill` 미적용 상태로 측정 — literature의 이득 조건을 완전히 재현하진 않음. 단 이는 **우리 운영 의사결정과 무관**(운영도 동일 기본 플래그).
- FP8 non-lossless로 NO-MTP/MTP 출력 길이 비대칭 — `tok/s`(정규화 지표)로 결론은 견고하나 TTLP 직접 비교는 제한적.
- A100는 FP8 연산 가속이 없어, H100 등 Hopper에서는 절대 수치가 다를 수 있음.

---

## 10. 부록

### 10.1 디렉토리 구조

```
test_mtp/
├── README.md            # 본 보고서
├── scripts/
│   ├── benchmark.py     # 모델 자동감지 + temp=0 + 동시성 + acceptance 측정
│   ├── start-llm-test.sh# gamma 파라미터화 vLLM 기동 래퍼
│   └── compare-mtp4.sh  # gamma 스윕 오케스트레이션 (메모리 정리 검증 포함)
└── raw_logs/
    ├── compare-mtp.log  # A3B v1 (샘플링)
    ├── compare-mtp2.log # A3B v2 (temp=0 + 동시성)
    ├── compare-mtp3.log # Dense 27B v3 (gamma=1)
    └── compare-mtp4.log # Dense 27B v4 (gamma 스윕 2/3)
```

### 10.2 vLLM 기동 (MTP on, gamma 가변)

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

MTP off는 `--speculative-config`만 제거.

### 10.3 acceptance rate 추출

```bash
curl -s http://localhost:8000/metrics | grep spec_decode_num
# acceptance = accepted_tokens_total / draft_tokens_total
```

### 10.4 재현 시 메모리 정리

페이즈 전환 시 `systemctl stop` 후 `nvidia-smi`로 `memory.used < 2000MB` 도달까지 대기 → GPU 완전 해제 보장 후 다음 페이즈.
