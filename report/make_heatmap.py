#!/usr/bin/env python3
"""1부 통합 히트맵(속도+정확도) 생성.

척도 전략:
  - 속도 4열(base S/8K, MTP S/8K): 같은 tok/s 단위 -> 공통 로그 척도(가로 비교 가능).
  - 정확도 6열: 단위/동적범위 제각각 -> 열별 min-max 독립 정규화(세로로만 읽음).
색은 두 그룹을 각각 0..1로 정규화해 하나의 순차 컬러맵으로 표현하고,
셀에는 실제 수치(tok/s, 0-1 정답률)를 적어 절대값도 함께 보이게 한다.

재현:  uv run --with matplotlib --with numpy report/make_heatmap.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 패밀리 정렬: Gemma(12,26,31,diff) -> Qwen(35,27)
N = None
rows = [
    ("Gemma 12B bf16",    [82, 78, 184, 148],   [0.662, 0.652, 0.788, 0.913, 0.832, 0.596]),
    ("Gemma 12B fp8",     [117, 110, 199, 182], [0.652, 0.616, 0.814, 0.916, 0.844, 0.610]),
    ("Gemma 12B qat",     [131, 114, N, N],     [0.636, 0.581, 0.782, 0.910, 0.826, 0.570]),
    ("Gemma 26B-A4B bf16", [200, 188, 322, 312], [0.763, 0.727, 0.870, 0.930, 0.890, 0.644]),
    ("Gemma 26B-A4B fp8",  [226, 212, 389, 336], [0.773, 0.707, 0.798, 0.919, 0.902, 0.626]),
    ("Gemma 26B-A4B int8", [218, 201, 382, 331], [0.758, 0.732, 0.824, 0.923, 0.890, 0.614]),
    ("Gemma 31B bf16",    [40, 35, N, N],       [0.828, 0.788, 0.840, 0.941, 0.904, 0.692]),
    ("Gemma 31B fp8",     [68, 62, 178, 150],   [0.849, 0.763, 0.862, 0.944, 0.898, 0.708]),
    ("Gemma 31B qat",     [90, 71, 214, 143],   [0.833, 0.773, 0.850, 0.943, 0.904, 0.688]),
    ("Gemma4 diff 26B-A4B fp8",  [864, 578, N, N], [0.571, 0.606, 0.706, 0.890, 0.850, 0.576]),
    ("Gemma4 diff 26B-A4B bf16", [616, 372, N, N], [0.596, 0.631, 0.738, 0.893, 0.848, 0.576]),
    ("Gemma4 diff 26B-A4B int8", [759, 510, N, N], [0.576, 0.672, 0.744, 0.897, 0.850, 0.544]),
    ("Qwen3.6 35B-A3B fp8", [217, 205, 311, 318], [0.803, 0.838, 0.862, 0.904, 0.834, 0.586]),
    ("Qwen3.6 27B bf16",  [48, 45, N, N],       [0.798, 0.848, 0.856, 0.939, 0.846, 0.628]),
    ("Qwen3.6 27B fp8",   [79, 74, 147, 142],   [0.828, 0.854, 0.846, 0.925, 0.850, 0.634]),
    ("Qwen3.6 27B int8",  [48, 45, N, N],       [0.818, 0.874, 0.868, 0.932, 0.854, 0.658]),
]
spd_cols = ["base S", "base 8K", "MTP S", "MTP 8K"]
acc_cols = ["lm-GPQA", "insp-GPQA", "MMLU-Pro", "IFEval", "haerae", "KMMLU"]
cols = spd_cols + acc_cols
nrow, ncol = len(rows), len(cols)

raw = np.full((nrow, ncol), np.nan)
for i, (_, s, a) in enumerate(rows):
    for j, v in enumerate(s):
        raw[i, j] = np.nan if v is None else v
    for j, v in enumerate(a):
        raw[i, 4 + j] = v

# 0..1 정규화 행렬
norm = np.full_like(raw, np.nan)
# 속도: 공통 로그 척도
sv = np.array([v for v in raw[:, :4].ravel() if not np.isnan(v)])
slo, shi = math.log10(sv.min()), math.log10(sv.max())
for i in range(nrow):
    for j in range(4):
        if not np.isnan(raw[i, j]):
            norm[i, j] = (math.log10(raw[i, j]) - slo) / (shi - slo)
# 정확도: 열별 min-max
for j in range(4, ncol):
    col = raw[:, j]
    lo, hi = np.nanmin(col), np.nanmax(col)
    norm[:, j] = (col - lo) / (hi - lo)

cmap = LinearSegmentedColormap.from_list("kblue", ["#f7fbff", "#9ecae1", "#3182bd", "#08306b"])
cmap.set_bad("#e8e8e8")  # NA 셀

fig, ax = plt.subplots(figsize=(11, 8.2))
masked = np.ma.masked_invalid(norm)
ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)

# 셀 주석: 실제 수치
for i in range(nrow):
    for j in range(ncol):
        v = raw[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", color="#999", fontsize=7)
            continue
        txt = f"{int(round(v))}" if j < 4 else f"{v:.2f}".lstrip("0")
        shade = norm[i, j]
        ax.text(j, i, txt, ha="center", va="center",
                color="white" if shade > 0.55 else "#222", fontsize=7.5)

ax.set_xticks(range(ncol))
ax.set_xticklabels(cols, fontsize=7.8)
ax.set_yticks(range(nrow))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# 그룹 경계선: 속도|정확도(세로), Gemma|Qwen(가로)
ax.axvline(3.5, color="#444", lw=2)
ax.axhline(11.5, color="#444", lw=2)
# 그룹 헤더 (수평 열 라벨 위로 충분히 띄움)
ax.text(1.5, -1.7, "Speed  (log, shared scale)", ha="center", fontsize=9.5, weight="bold")
ax.text(6.5, -1.7, "Accuracy  (per-column scale)", ha="center", fontsize=9.5, weight="bold")
ax.text(-4.6, 5.5, "Gemma 4", rotation=90, va="center", fontsize=9.5, weight="bold")
ax.text(-4.6, 13.5, "Qwen 3.6", rotation=90, va="center", fontsize=9.5, weight="bold")

ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title("MTP / diffusion bench  —  speed + accuracy heatmap   (single H100 80GB)\n"
             "darker = higher  ·  speed: shared log scale (compare across)  ·  "
             "accuracy: per-column scale (compare down only)",
             fontsize=10.5, pad=52)
fig.tight_layout()
out = "report/heatmap.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("saved", out)
