#!/usr/bin/env python3
"""Figure 1: DeepMiRT architecture diagram."""

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent / "fig1_architecture.pdf"

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis("off")

# ── Colors ──
C_INPUT = "#dce6f1"
C_ENCODER = "#4a90d9"
C_CROSS = "#e67e22"
C_POOL = "#8e44ad"
C_MLP = "#27ae60"
C_OUTPUT = "#c0392b"

def draw_box(ax, xy, w, h, text, color, fontsize=8, text_color="white", alpha=1.0):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="none", alpha=alpha,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=text_color, fontweight="bold")
    return (cx, cy)

def draw_arrow(ax, start, end, color="black"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=1.2, connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)

# ── Input boxes ──
m_c = draw_box(ax, (0.3, 4.2), 2.0, 0.8, "miRNA\n(18–25 nt)", C_INPUT, fontsize=8, text_color="#333")
t_c = draw_box(ax, (0.3, 1.5), 2.0, 0.8, "Target\n(40 nt)", C_INPUT, fontsize=8, text_color="#333")

# ── RNA-FM Encoder ──
enc_m = draw_box(ax, (3.2, 4.2), 2.0, 0.8, "RNA-FM\nEncoder", C_ENCODER, fontsize=8)
enc_t = draw_box(ax, (3.2, 1.5), 2.0, 0.8, "RNA-FM\nEncoder", C_ENCODER, fontsize=8)

# Shared weights annotation
ax.annotate(
    "", xy=(3.7, 2.3), xytext=(3.7, 4.2),
    arrowprops=dict(arrowstyle="<->", color="#666", linewidth=1.0, linestyle="--"),
)
ax.text(3.05, 3.25, "shared\nweights", ha="center", va="center", fontsize=6.5, color="#666", style="italic")

# Embedding annotations
ax.text(5.35, 4.9, "(B, M, 640)", ha="left", va="center", fontsize=6, color="#888", family="monospace")
ax.text(5.35, 2.2, "(B, T, 640)", ha="left", va="center", fontsize=6, color="#888", family="monospace")

# ── Cross-Attention ──
ca_c = draw_box(ax, (6.0, 2.6), 2.0, 1.4, "Cross-\nAttention\n2 layers, 8 heads", C_CROSS, fontsize=7.5)

# Annotations for Q, K/V
ax.text(5.65, 3.7, "K, V", ha="center", va="center", fontsize=7, color=C_CROSS, fontweight="bold")
ax.text(5.65, 2.8, "Q", ha="center", va="center", fontsize=7, color=C_CROSS, fontweight="bold")

# ── Masked Mean Pool ──
pool_c = draw_box(ax, (6.3, 1.0), 1.4, 0.65, "Mean Pool", C_POOL, fontsize=7.5)

# ── MLP Head ──
mlp_c = draw_box(ax, (6.0, 0.0), 2.0, 0.65, "MLP  640→256→64→1", C_MLP, fontsize=7)

# ── Output ──
out_c = draw_box(ax, (8.6, 0.0), 1.1, 0.65, "P(bind)", C_OUTPUT, fontsize=8)

# ── Arrows ──
draw_arrow(ax, (2.3, 4.6), (3.2, 4.6))
draw_arrow(ax, (2.3, 1.9), (3.2, 1.9))
draw_arrow(ax, (5.2, 4.6), (6.0, 3.9))       # miRNA emb -> cross-attn (K,V)
draw_arrow(ax, (5.2, 1.9), (6.0, 2.8))        # target emb -> cross-attn (Q)
draw_arrow(ax, (7.0, 2.6), (7.0, 1.65))       # cross-attn -> pool
draw_arrow(ax, (7.0, 1.0), (7.0, 0.65))       # pool -> MLP
draw_arrow(ax, (8.0, 0.32), (8.6, 0.32))      # MLP -> output

# ── Training phase annotations ──
ax.text(
    5.0, 0.5, "Phase 1: freeze backbone\nPhase 2: unfreeze top 3 layers",
    ha="center", va="center", fontsize=6.5, color="#555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8),
)

# ── Title ──
ax.text(5.0, 5.3, "DeepMiRT Architecture", ha="center", va="center", fontsize=12, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
