#!/usr/bin/env python3
"""Redraw figures/architecture.png (LANet forward pass)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "architecture.png"
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, text, fc="#E8F0FA", ec="#2F5D9B", fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#1b1b1b", linespacing=1.4)


def arrow(x1, y1, x2, y2, c="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1, color=c, shrinkA=0, shrinkB=0))


def lab(x, y, t, c="#555", fs=7.2):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=c, style="italic")


ax.text(0.5, 54, "LANet (Lesion-Attention Network): forward pass", fontsize=12, fontweight="bold", va="center")
ax.text(0.5, 51, "about 3.1 M parameters and 0.23 GMACs at 224×224 (counted by the script)", fontsize=8, color="#444", va="center")
box(0.5, 33, 11, 8, "Input\n224×224×3", fc="#F3F3F3", ec="#777")
box(15, 29, 19, 16, "MobileNetV3-Large\nImageNet-pretrained\n(classifier head\nremoved)", fc="#DDEBD5", ec="#4C8C3A", bold=True)
arrow(11.7, 37, 14.8, 37)
box(38, 39.5, 15, 7.5, "F1 (stride 16)\n14×14×112"); box(38, 28, 15, 7.5, "F2 (stride 32)\n7×7×960")
arrow(34.2, 40.5, 37.8, 43.2); arrow(34.2, 33.5, 37.8, 31.8)
box(57, 39.5, 18, 7.5, "1×1 proj → 128\n+ depthwise 3×3\n(local detail)", fs=7.4)
box(57, 28, 18, 7.5, "1×1 proj → 128\n+ upsample to 14×14\n(global context)", fs=7.4)
arrow(53.2, 43.25, 56.8, 43.25); arrow(53.2, 31.75, 56.8, 31.75)
box(79, 29.5, 20.5, 14, "Gated fusion\nfused = local ⊙ gate + global\ngate = σ(1×1 conv(global))\n14×14×128", fc="#FCE4E4", ec="#B03A3A", fs=7.2)
arrow(75.2, 42.8, 78.8, 41.0); arrow(75.2, 32.5, 78.8, 33.5)
box(79, 12.5, 20.5, 11, "Attention head\n1×1 (128→32) → SiLU\n→ 1×1 (→1) → σ\nA: 14×14×1", fc="#EADCF5", ec="#6B3FA0", fs=7.4)
arrow(89.25, 29.3, 89.25, 23.7)
box(38, 12.5, 32, 11, "Attention pooling: Σ(w · fused),  w = A / ΣA\n+ average pooling of fused\n→ concatenated vector (256)", fs=7.6)
arrow(78.8, 18, 70.2, 18, c="#6B3FA0"); lab(74.5, 20.3, "A as weights", c="#6B3FA0")
arrow(83, 29.3, 66, 23.8); lab(79.5, 25.6, "fused features")
box(38, 1.5, 32, 7, "Dropout 0.2 → FC (256 → 5)\nCBB | CBSD | CGM | CMD | Healthy", fc="#F3F3F3", ec="#777", fs=7.6)
arrow(54, 12.3, 54, 8.7)
lab(89.25, 8.5, "A also gives the noise module its focus\nscore; penalty 0.05 · mean(A) in the loss", c="#6B3FA0", fs=7.2)
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
