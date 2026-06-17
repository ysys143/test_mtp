# Phase 2 결과 — 컨텍스트 길이 안정화

> **[통합] 본 문서는 종합본 `report/`로 대체됨 → [../report/README.md](../report/README.md) (컨텍스트는 report/03-D, 배선은 01).** 아래는 원본 보존.

**측정:** 단일 H100 80GB, util 0.95, max-num-seqs 1. probe(=vLLM KV 리포트/OOM estmax) + NIAH(needle 회수).
**데이터:** `max_context.csv` · **스크립트:** `probe_maxlen.sh`, `niah.py`

## 1. 핵심 결론: 8K는 한계가 아니었다 — 전 셀 네이티브 256K 도달 가능

모든 모델×정밀도가 **네이티브 262,144(256K)** 컨텍스트에 도달한다. 8K는 실한계의 **1/32**에 불과했다.
NIAH로 "로드 ≠ 작동"을 검증 — 241K 컨텍스트에서 깊이 10/50/90% 전부 회수 성공.

| 모델 | 정밀도 | 최대 안정 컨텍스트 | 비고 |
|---|---|---|---|
| 26B-A4B (MoE) | bf16/fp8/int8 | **256K** | KV 0.97~1.85M 토큰 (sliding-window) |
| 31B (dense) | bf16 | 94K → **256K** | auto-KV는 94K, **`--kv-cache-dtype fp8`로 256K** |
| 31B (dense) | fp8 | **256K** | auto-KV로 도달 |
| 31B (dense) | qat-w4a16 | **256K** | auto-KV로 도달 |
| 12B (dense) | bf16/fp8 | **256K** | KV 1.96~2.35M 토큰 |
| Qwen35-A3B (MoE + gated-delta) | fp8 | **256K** | 선형어텐션 3:1 + MoE; full층 GQA |
| Qwen27 (gated-delta, dense) | fp8 | **256K** | 선형어텐션 3:1; `--max-num-seqs ≤ 783` |

## 2. 왜 이렇게 길게 들어가나 (KV 구조)
- **Gemma4: `sliding_attention`×3 + `full_attention`×1** (`sliding_window=1024`) — 대부분 레이어가 1024 윈도우만 유지, 글로벌(full) 레이어만 전역 → KV가 컨텍스트에 거의 안 늘어남. NIAH로 글로벌 레이어의 241K 전역 회수 확인(3/3). **mamba/gated-delta 아님.**
- **Qwen3.6: 두 모델 다 gated-delta 선형어텐션 하이브리드** (`linear_attention`×3 + `full_attention`×1, `full_attention_interval=4`, `mamba_ssm_dtype`/`linear_conv_kernel_dim` 존재) — 전체의 1/4 층만 full-attention(나머지 선형층은 KV≈0, SSM 상태만), 그 full 층도 GQA → 토큰당 KV가 작음. 35B는 여기에 MoE가 추가될 뿐, 선형어텐션 구조는 27B와 동일.

## 3. KV-dtype fp8 = 무거운 dense bf16의 컨텍스트 레버
31B-bf16(62GB 가중치)만 auto-KV로 ~94K에서 OOM. `--kv-cache-dtype fp8`로 KV를 절반 내리면 **풀 256K**.
NIAH 결과 fp8 KV가 장거리 회수를 **열화시키지 않음**(31B-bf16+kvfp8 241K 3/3). → 무거운 bf16 셀의 표준 설정으로 채택.

## 4. int8 weight-only는 MoE 전용 (dense는 no-op)
`int8_per_channel_weight_only`의 quantization_config = **`linear=None, moe=int8`** (vLLM 로그 증거).
- 26B-A4B(MoE): 전문가 양자화 → KV 0.97M→1.79M (효과)
- 12B/31B(dense): 가중치 22.83GiB=bf16 (no-op) → dense int8 = bf16
→ **dense의 저정밀 축은 fp8 + qat-w4a16**(원본 설계와 정합). dense int8은 별도 W8A8 오프라인 체크포인트가 있어야 가능(현 범위 밖).

## 5. NIAH 검증 (max_tokens=512 통일)
대표 5셀 × 깊이 3 = 15회 전부 회수 성공. 직교 인자(아키텍처 3종 × KV-dtype 2종) 전부 커버:

| 대표 셀 | 아키텍처 / KV | 결과 |
|---|---|---|
| 26B-fp8 | Gemma MoE sliding-window / auto | 3/3 @241K |
| 12B-fp8 | Gemma dense / auto | 3/3 @241K |
| 31B-bf16+kvfp8 | Gemma dense / **fp8 KV** | 3/3 @241K |
| qwen35-fp8 | Qwen MoE + gated-delta(3:1) / auto | 3/3 @241K |
| qwen27-fp8 | Qwen **gated-delta(3:1) dense** / auto | 3/3 @241K |

> 주의: Qwen은 추론 서두를 붙여 max_tokens가 작으면(32) 답이 잘려 0/3로 보였음 — 512로 통일해 해결.
> 정확도/NIAH 등 생성-채점은 "가장 말 많은 모델" 기준 출력 예산으로 통일해야 공정.

## 6. Phase 3용 고정 설정
- **서빙**: `--max-model-len 262144`(또는 평가 길이), util 0.95.
- **31B-bf16/int8**: `--kv-cache-dtype fp8` 필수(아니면 94K cap).
- **Qwen27**: v0.22.1 이미지 + `--max-num-seqs ≤ 783`.
- **공통 비교 길이**: 전 셀 256K 가능하나, 평가 배터리(RULER 등)는 16K/32K/64K/128K 티어로 운용 가능.
- **생성 채점 출력 예산**: max_tokens ≥ 512 (Qwen 추론 서두 대응).
