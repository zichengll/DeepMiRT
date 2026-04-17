#!/usr/bin/env python3
"""Figure 3: Test set performance — ROC, PR, and calibration panels."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# ── Paths ──
PRED_FILE = pathlib.Path(__file__).resolve().parents[2] / "evaluation_outputs" / "predictions_test.csv"
OUT = pathlib.Path(__file__).resolve().parent / "fig3_performance.pdf"

# ── Load predictions (subsample for speed if needed) ──
print("Loading predictions...")
df = pd.read_csv(PRED_FILE, usecols=["label", "prob"])
labels = df["label"].values
probs = df["prob"].values

# ── Compute curves ──
print("Computing ROC curve...")
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

print("Computing PR curve...")
precision, recall, _ = precision_recall_curve(labels, probs)
pr_auc = auc(recall, precision)

print("Computing calibration curve...")
fraction_pos, mean_predicted = calibration_curve(labels, probs, n_bins=15, strategy="uniform")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

# (A) ROC Curve
ax = axes[0]
ax.plot(fpr, tpr, color="#2171b5", linewidth=1.8, label=f"DeepMiRT (AUROC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6)
ax.set_xlabel("False Positive Rate", fontsize=9)
ax.set_ylabel("True Positive Rate", fontsize=9)
ax.set_title("(A) ROC Curve", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")

# (B) PR Curve
ax = axes[1]
baseline = labels.mean()
ax.plot(recall, precision, color="#e6550d", linewidth=1.8, label=f"DeepMiRT (AUPRC = {pr_auc:.4f})")
ax.axhline(y=baseline, linestyle="--", color="gray", linewidth=0.8, alpha=0.6, label=f"Baseline ({baseline:.2f})")
ax.set_xlabel("Recall", fontsize=9)
ax.set_ylabel("Precision", fontsize=9)
ax.set_title("(B) Precision-Recall Curve", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1.05)

# (C) Calibration Reliability
ax = axes[2]
ax.plot(mean_predicted, fraction_pos, "s-", color="#2ca02c", markersize=5, linewidth=1.5, label="DeepMiRT")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6, label="Perfectly calibrated")

# Add histogram of predictions
ax2 = ax.twinx()
ax2.hist(probs, bins=50, range=(0, 1), alpha=0.15, color="#2ca02c", density=True)
ax2.set_ylabel("Density", fontsize=8, color="gray", alpha=0.6)
ax2.tick_params(axis="y", labelsize=7, colors="gray")

ax.set_xlabel("Mean Predicted Probability", fontsize=9)
ax.set_ylabel("Fraction of Positives", fontsize=9)
ax.set_title("(C) Calibration (ECE = 0.0064)", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

fig.tight_layout(w_pad=2.5)
fig.savefig(OUT, bbox_inches="tight", dpi=300)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
