"""Multi-seed significance analysis for D1/A1 ablations.

Parses checkpoint filenames in checkpoints/ablation/<exp>_seed<N>/,
runs paired t-test and Wilcoxon signed-rank against baseline, emits
CSV + markdown.

Spec: docs/superpowers/specs/2026-04-17-multiseed-significance-design.md
"""
from __future__ import annotations

import re
from pathlib import Path

CKPT_RE = re.compile(r"epoch=\d+-val_auroc=([0-9.]+)\.ckpt$")


def best_val_auroc(run_dir: Path) -> float:
    """Return the maximum val_auroc parsed from epoch=*-val_auroc=*.ckpt files.

    Ignores last.ckpt and any other non-matching files.
    Raises FileNotFoundError if no matching checkpoint files exist.
    """
    scores = []
    for p in run_dir.iterdir():
        m = CKPT_RE.match(p.name)
        if m:
            scores.append(float(m.group(1)))
    if not scores:
        raise FileNotFoundError(f"No epoch=*-val_auroc=*.ckpt in {run_dir}")
    return max(scores)


import math
from typing import Sequence

import numpy as np
from scipy import stats


def paired_stats(baseline: Sequence[float], ablation: Sequence[float]) -> dict:
    """Paired comparison of ablation vs baseline (per-seed differences).

    Returns dict with: n, mean_diff, std_diff, ci95_low, ci95_high,
    t_stat, p_ttest, w_stat, p_wilcoxon, cohens_dz.

    Differences are defined as ablation - baseline (positive = ablation
    better). Wilcoxon is run with zero_method='wilcox' (drop zeros);
    if all diffs are zero or n<3, p_wilcoxon is NaN.
    """
    if len(baseline) != len(ablation):
        raise ValueError(
            f"length mismatch: baseline={len(baseline)} ablation={len(ablation)}"
        )
    n = len(baseline)
    diffs = np.asarray(ablation, dtype=float) - np.asarray(baseline, dtype=float)
    mean_d = float(diffs.mean())
    std_d = float(diffs.std(ddof=1)) if n > 1 else 0.0

    # paired t-test
    t_res = stats.ttest_rel(ablation, baseline)
    t_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else math.nan
    p_t = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else math.nan

    # 95% CI for mean difference (t-distribution, df=n-1)
    if n > 1 and std_d > 0:
        sem = std_d / math.sqrt(n)
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=sem)
        ci_low, ci_high = float(ci_low), float(ci_high)
    else:
        ci_low = ci_high = mean_d

    # Wilcoxon signed-rank (paired non-parametric)
    if n >= 1 and np.any(diffs != 0):
        try:
            w_res = stats.wilcoxon(ablation, baseline, zero_method="wilcox")
            w_stat = float(w_res.statistic)
            p_w = float(w_res.pvalue)
        except ValueError:
            w_stat = math.nan
            p_w = math.nan
    else:
        w_stat = math.nan
        p_w = math.nan

    # Cohen's d_z (paired)
    if std_d > 0:
        dz = mean_d / std_d
    elif mean_d == 0:
        dz = 0.0
    else:
        # perfect consistency (zero variance in diffs) → infinite effect size
        dz = math.copysign(math.inf, mean_d)

    return {
        "n": n,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": t_stat,
        "p_ttest": p_t,
        "w_stat": w_stat,
        "p_wilcoxon": p_w,
        "cohens_dz": dz,
    }
