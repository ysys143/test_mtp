# 결과 — 종합 매트릭스

**환경:** 단일 H100 80GB (GCP), vLLM `nightly`(AR/MTP) · `:gemma`(diffusion)
**측정:** non-streaming 총처리량(출력 512토큰 고정 / wall-clock, prefix-cache off), MTP=γ4(Gemma)·γ2(Qwen)
**정확도:** ops 근본원인 식별 태스크, N=30, 채점기 `bench_acc2`(정의: `02-methodology.md`)
**측정일:** 2026-06-11 ~ 06-14 · 수치 검증: `../data/results.csv`

---

## 처리량 (tok/s, short context [8K context])

| 모델 (활성 param) | 정밀도 | AR base | AR + MTP (가속) | diffusion |
|---|---|---|---|---|
| **Gemma 26B-A4B** (MoE ~4B) | BF16 | 199.8 [187.7] | 322.9 [312.9] (1.62x) | 615.5 |
| | fp8 | 226.0 [211.6] | 390.3 [337.9] (1.73x) | **864.4** |
| | int8 | 218.2 [200.9] | 381.6 [331.2] (1.75x) | 618.4 [573.3] |
| **Gemma 31B** (dense) | BF16 | 40.5 [37.8] | — (OOM¹) | — |
| | fp8 | 67.4 [61.8] | 178.1 [150.2] (**2.64x**) | — |
| | qat w4a16 | 89.6 [70.7] | 213.7 [143.1] (2.38x) | — |
| **Gemma 12B** (dense) | BF16 | 82.4 [78.1] | 186.5 [157.9] (2.26x) | — |
| | fp8 | 118.7 [111.6] | 200.7 [178.6] (1.69x) | — |
| **Qwen3.6 35B-A3B** (MoE ~3B) | fp8 | 216.7 [204.0] | 315.6 (γ2) · 240.4 (γ1) (1.46x) | n/a |
| **Qwen3.6 27B** (Mamba-hybrid²) | fp8 | 79.2 [73.8] | 146.9 [141.5] (1.85x) | n/a |

## 정확도 (N=30, ops 근본원인 태스크)

| 모델 | 정밀도 | AR / base | diffusion |
|---|---|---|---|
| Gemma 26B-A4B | BF16 / fp8 / int8 | 100% / 100% / 100% | 86.7% / 90.0% / 93.3% ⁴ |
| Gemma 31B | BF16 / fp8 / qat | 100% / 100% / 100% | — |
| Gemma 12B | BF16 / fp8 | 100% / 100% | — |
| Qwen3.6 35B-A3B | fp8 | 83.3%³ | — |
| Qwen3.6 27B | fp8 (Mamba-hybrid²) | 90.0% | — |

> **천장 효과 + 변별력 주의:** Gemma 전 셀이 100%로 포화돼 있다. 근본원인이 로그에 명시 신호(`OOMKilled`
> 등 경성 에러 + dependents의 `upstream timeout calling {root}`가 root를 직접 호명)되어 Gemma엔 쉽기 때문.
> 단 **Qwen은 83~90%로 100% 미달** → 태스크가 완전히 자명하진 않고 어려운 끝에서 변별은 한다. 그래도
> Gemma 포화 탓에 이 태스크로 **Gemma 정밀도 간 미세 우열은 못 가린다**(양자화 무손실은 "현저한 손상
> 없음"의 하한 보증). 정밀 변별엔 더 어려운(천장에 안 닿는) 태스크 필요 — 후속(`../PLAN.md` Phase 3).

---

## 각주 — 불가/예외 셀 (gap 아님, 결론으로 문서화)

1. **31B-bf16-MTP**: 62GB(bf16 가중치)가 80GB의 78%를 차지 → 드래프터+KV+cudagraph 동시 수용 불가.
   cudagraph 켜면 KV<6.89GB로 OOM, `--enforce-eager`로만 기동되나 그 수치는 커널런치 오버헤드로
   비교 불가. **단일 H100 80GB에서 깨끗한 측정 불가**로 결론. fp8/qat에선 MTP 정상(2.64x/2.38x).
   상세: `04-troubleshooting.md` #8.
2. **Qwen27-FP8 = Mamba-하이브리드(SSM+attention), dense 아님.** 측정 조건: (a) stable
   `vllm/vllm-openai:v0.22.1` 이미지, (b) **`--max-num-seqs ≤ 783`**(기본 1024가 Mamba 캐시 블록 초과 →
   cudagraph 캡처 실패). MTP는 init이 느려(드래프터+mamba/attention page 정렬+cudagraph, ~366s) 헬스
   타임아웃을 넉넉히 줘야 함. γ2 기준. 이전 "DIED/비호환"은 전부 이 설정·타임아웃 문제였음 —
   아키텍처 비호환 아님(`04-troubleshooting.md`).
3. **Qwen 정확도는 수정 채점기 `bench_acc2`로 측정**(max_tokens 24→256 + "마지막 언급 서비스명" 추출).
   초기 6.7%는 `bench_acc.py`의 max_tokens=24가 Qwen 추론 서두를 답 출력 전에 잘라낸 아티팩트.
   신규 채점기로 **Gemma 26B-fp8 control = 100% 재확인(공정성 검증)** → Qwen35 83.3%, Qwen27 90.0%는
   실측. Gemma 표값(100%)은 두 채점기에서 동일. 상세: `04-troubleshooting.md` #10.
4. **diffusion 변형 간 정확도 차이는 노이즈 — 순위 무의미.** 86.7/90.0/93.3%는 분수로 **26/27/28 of 30**,
   정밀도 단계당 딱 1문제 차. N=30, p≈0.9의 95% 신뢰구간 ≈ ±11%p라 셋이 통째로 겹침. "양자화가 정확도를
   올린다"는 물리적으로 기대 안 되는 방향이며 소표본 아티팩트. 말할 수 있는 건 "diffusion 3종 모두 AR(30/30)
   보다 낮다"까지. (N=30은 미세 정확도 비교에 부족 — Qwen35 25/30 vs Qwen27 27/30도 동일 한계. `../PLAN.md` Phase 3에서 N↑.)

> 기법 해석은 `03-findings.md`, 측정 신뢰 근거는 `02-methodology.md`.
> 정정 이력(streaming 결함으로 무효화된 초기 결론 등)은 `../legacy/` 보존.
