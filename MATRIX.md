# 평가 매트릭스 — 3 프레임워크 × 전 모델 (GOAL 고정)

> **[통합] 본 계획/매트릭스는 종합본 `report/`로 대체됨 → [report/README.md](report/README.md) (커버리지는 README, 배선은 01-프로토콜).** 아래는 원본 보존.

> **GOAL**: 요청한 **모든 모델**에 (lm-eval / inspect_ai / HRET) **세 프레임워크를 전부 올바르게** 적용.
> **올바른 적용 = 동등성 조건 전부**: ① thinking ON (enable_thinking=true) ② 인위적 답변길이 제약 금지
> (인프라 한계만 — 관대한 max_tokens로 자연완결, 절단 0) ③ 동일 샘플링(temp0.6/top_p0.95) ④ 동일 프롬프트.
> 지표: 정확도(3 프레임워크 배터리) + 속도(tok/s) + 컨텍스트/메모리.

## 모델 (요청 전체)
Gemma 12B(dense) · 26B-A4B(MoE) · 31B(dense) · **diffusiongemma-26B-A4B** · Qwen3.6 27B · 35B-A3B(MoE)
정밀도 sub-axis: bf16 · fp8 · int8(MoE/diffusion 실효, dense=no-op) · qat-w4a16(Gemma 12B/31B)
기법 sub-axis: AR · MTP · Diffusion · Thinking

## 세 프레임워크 = 다면 배터리
| 프레임워크 | 상태 | 벤치마크(적용) | 커버리지 |
|---|---|---|---|
| **lm-eval** 0.4.12 | 설치됨 | MMLU-Pro(full) · GPQA-Diamond(full) | 영어 지식·복합추론 |
| **inspect_ai** + inspect_evals | 설치중 | gpqa_diamond · mmlu_pro · ifeval(지시따르기) (+ math/gsm8k) | 추론 + 지시준수 |
| **HRET** (haerae-evaluation-toolkit) | 설치중 | KMMLU · HAE-RAE Bench · KoBEST (+영어) | 한국어 |

> 세 프레임워크는 **서로 다른 면**을 봄(영어지식 / 지시준수 / 한국어) → "다면 평가"의 실체.
> 셋 다 vLLM OpenAI 엔드포인트 타격 → **inject_proxy(enable_thinking=true) 경유로 thinking-on 통일**, 각 프레임워크 max_tokens=32768(절단 0).

## 동등조건 배선 (전 프레임워크 공통)
- serve: vLLM `--reasoning-parser {gemma4|qwen3}` (thinking을 content서 분리) + 정밀도 flag.
- thinking-on: 요청에 `chat_template_kwargs.enable_thinking=true` 주입(inject_proxy :8001). 세 클라가 전부 이걸 경유.
- 길이: max_tokens=32768 (자연완결, 절단 검증 finish_reason=length≈0). **예산서 강제 종료 금지.**
- 샘플링: temperature=0.6, top_p=0.95 통일.

## 실행 순서 (전 셀, 재개가능·셀별 즉시저장 — Spot 대비)
- **R0 셋업**: inspect_ai/HRET 설치 → 각 프레임워크 스모크(1모델, thinking-on, 절단0 확인).
- **R1 코어**: 6 모델 × 대표정밀도(fp8/native) × thinking-on × **3 프레임워크 전부**. (다면 비교 본체)
- **R2 정밀도**: bf16/int8/qat × 3 프레임워크.
- **R3 기법**: MTP(속도+lossless) · Diffusion(:gemma, 3정밀도 × 3 프레임워크 + 속도).
- **R4 속도/컨텍스트**: 전 셀 throughput + diffusion/Qwen 컨텍스트 보강.
- **R5 합의**: 통합 CSV + 트레이드오프 프론티어(속도×정확도×메모리) → REPORT.

## 이미 측정됨 (재사용 — 동일 하네스)
- lm-eval MMLU-Pro(no-think): 12B/26B/31B 각 정밀도 + Qwen fp8 (예: 12B-fp8=0.771)
- lm-eval GPQA think(fp8): 26B 0.7626 · 31B 0.8535 · 12B 0.6667 · qwen35 0.7929 · qwen27(완료)
- lm-eval GPQA no-think 정밀도 grid: 26B/31B/12B/Qwen (Section B)
- 절단검증: 31B 50/50 stop(length=0) 통과
- 컨텍스트: Gemma 전정밀도 + Qwen fp8 = 256K
> 이들은 thinking-on/관대길이 조건으로 재사용 가능분만 채택, 조건 불일치분은 R1~R3에서 재측정.
