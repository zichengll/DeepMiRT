#!/usr/bin/env python3
"""Figure 2: miRBench benchmark comparison — grouped bar chart."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──
TABLE_DIR = pathlib.Path(__file__).resolve().parents[2] / "evaluation_outputs" / "tables"
OUT = pathlib.Path(__file__).resolve().parent / "fig2_benchmark.pdf"

# ── Load data ──
datasets = {
    "CLASH\n(Hejret 2023)": "benchmark_AGO2_CLASH_Hejret2023.csv",
    "eCLIP\n(Klimentova 2022)": "benchmark_AGO2_eCLIP_Klimentova2022.csv",
    "eCLIP\n(Manakov 2022)": "benchmark_AGO2_eCLIP_Manakov2022.csv",
}

# Top 8 methods to display (by mean AUROC across datasets)
all_dfs = {}
for label, fname in datasets.items():
    df = pd.read_csv(TABLE_DIR / fname)
    df["Method"] = df["Method"].str.replace(r"_\w+\d{4}$", "", regex=True)
    df["Method"] = df["Method"].replace("Ours (RNA-FM)", "DeepMiRT")
    all_dfs[label] = df.set_index("Method")["AUROC"]

combined = pd.DataFrame(all_dfs)
mean_auroc = combined.mean(axis=1).sort_values(ascending=False)
top_methods = mean_auroc.head(8).index.tolist()

# ── Plot ──
fig, ax = plt.subplots(figsize=(7.5, 3.8))

n_datasets = len(datasets)
n_methods = len(top_methods)
x = np.arange(n_datasets)
width = 0.09
offsets = np.linspace(-(n_methods - 1) / 2 * width, (n_methods - 1) / 2 * width, n_methods)

colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
ours_color = "#2171b5"

dataset_labels = list(datasets.keys())

for i, method in enumerate(top_methods):
    vals = [combined.loc[method, ds] if method in combined.index else 0.5 for ds in dataset_labels]
    color = ours_color if method == "DeepMiRT" else colors[i]
    edgecolor = "black" if method == "DeepMiRT" else "none"
    lw = 1.2 if method == "DeepMiRT" else 0
    bars = ax.bar(
        x + offsets[i], vals, width,
        label=method, color=color, edgecolor=edgecolor, linewidth=lw,
        zorder=3 if method == "DeepMiRT" else 2,
    )

# Add #1 labels on eCLIP bars for DeepMiRT
ours_idx = top_methods.index("DeepMiRT")
for ds_idx, ds_label in enumerate(dataset_labels):
    if "eCLIP" in ds_label:
        val = combined.loc["DeepMiRT", ds_label]
        ax.annotate(
            "#1", (x[ds_idx] + offsets[ours_idx], val),
            textcoords="offset points", xytext=(0, 4),
            ha="center", va="bottom", fontsize=7, fontweight="bold", color=ours_color,
        )

ax.set_xticks(x)
ax.set_xticklabels(dataset_labels, fontsize=9)
ax.set_ylabel("AUROC", fontsize=10)
ax.set_ylim(0.48, 0.84)
ax.set_title("miRBench Standard Benchmark Comparison", fontsize=11, fontweight="bold", pad=10)
ax.legend(
    fontsize=6.5, ncol=4, loc="upper center",
    bbox_to_anchor=(0.5, -0.12), frameon=False, columnspacing=1,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
