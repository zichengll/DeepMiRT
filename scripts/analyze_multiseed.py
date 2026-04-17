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


import argparse
import csv
from collections import defaultdict


def collect_results(
    ckpt_root: Path,
    configs: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, dict[int, float]]:
    """Walk checkpoints/ablation/<config>_seed<S>/ and return nested dict.

    Skips (config, seed) pairs whose dir does not exist or is empty —
    prints a warning so the caller knows what was excluded.
    """
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for cfg in configs:
        for s in seeds:
            run_dir = ckpt_root / f"{cfg}_seed{s}"
            if not run_dir.is_dir():
                print(f"[WARN] missing: {run_dir}")
                continue
            try:
                out[cfg][s] = best_val_auroc(run_dir)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
    return out


def write_csv(results: dict[str, dict[int, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "val_auroc"])
        for cfg, by_seed in results.items():
            for s in sorted(by_seed):
                w.writerow([cfg, s, f"{by_seed[s]:.6f}"])


def write_markdown(
    results: dict[str, dict[int, float]],
    baseline_key: str,
    ablations: Sequence[str],
    seeds: Sequence[int],
    path: Path,
    alpha: float = 0.025,
) -> None:
    """Write paper/multi_seed_significance.md with summary table + interp."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # paired vectors over seeds present in BOTH baseline and ablation
    def paired(cfg: str) -> tuple[list[float], list[float], list[int]]:
        b = results.get(baseline_key, {})
        a = results.get(cfg, {})
        common = sorted(set(b) & set(a))
        return [b[s] for s in common], [a[s] for s in common], common

    lines = []
    lines.append("# Multi-Seed Significance: D1 (uniform_lr) and A1 (concat)\n")
    lines.append(f"**Baseline:** `{baseline_key}` (default.yaml under ablation framework)  ")
    lines.append(f"**Seeds requested:** {list(seeds)}  ")
    lines.append(f"**Significance threshold:** α = {alpha} (Bonferroni-corrected for 2 tests)  ")
    lines.append(f"**Code pin:** git tag `ablation-pin-2026-04`\n")

    lines.append("## Per-seed val_auroc\n")
    lines.append("| seed | " + " | ".join([baseline_key] + list(ablations)) + " |")
    lines.append("|---|" + "|".join(["---"] * (1 + len(ablations))) + "|")
    for s in sorted(seeds):
        row = [str(s)]
        for cfg in [baseline_key] + list(ablations):
            v = results.get(cfg, {}).get(s)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Paired statistics vs baseline\n")
    lines.append("| ablation | n | mean Δ (pp) | 95% CI (pp) | paired t p | Wilcoxon p | Cohen's d_z | sig at α=" + f"{alpha}" + " |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cfg in ablations:
        b, a, common = paired(cfg)
        if len(b) < 2:
            lines.append(f"| {cfg} | {len(b)} | — | — | — | — | — | insufficient data |")
            continue
        st = paired_stats(b, a)
        sig = "✅" if (not math.isnan(st["p_ttest"]) and st["p_ttest"] < alpha) else "❌"
        lines.append(
            f"| {cfg} | {st['n']} | {st['mean_diff']*100:+.3f} | "
            f"[{st['ci95_low']*100:+.3f}, {st['ci95_high']*100:+.3f}] | "
            f"{st['p_ttest']:.4g} | {st['p_wilcoxon']:.4g} | {st['cohens_dz']:+.2f} | {sig} |"
        )
    lines.append("")

    lines.append("## Methods note\n")
    lines.append(
        "Each seed varies three random sources jointly: data shuffling order, "
        "model weight initialization, and DDP allreduce ordering under bf16. "
        "We do not enable `torch.use_deterministic_algorithms(True)` because the "
        "~20% throughput cost is not justified at this experiment scale; the "
        "remaining non-determinism is folded into the seed-to-seed variance "
        "estimate. n=5 is at the lower end of paired ablation reporting; for "
        "borderline effects we report `p_wilcoxon` as a non-parametric sanity "
        "check but do not include it in the Bonferroni family.\n"
    )

    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-root", type=Path,
        default=Path("checkpoints/ablation"),
    )
    ap.add_argument(
        "--seeds", type=str, default="42,123,456,789,2024",
        help="comma-separated seed list",
    )
    ap.add_argument(
        "--baseline", type=str, default="baseline",
    )
    ap.add_argument(
        "--ablations", type=str, default="D1_uniform_lr,A1_concat",
        help="comma-separated ablation config names",
    )
    ap.add_argument("--csv-out", type=Path, default=Path("paper/multi_seed_raw.csv"))
    ap.add_argument("--md-out", type=Path, default=Path("paper/multi_seed_significance.md"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ablations = args.ablations.split(",")
    configs = [args.baseline] + ablations

    results = collect_results(args.ckpt_root, configs, seeds)
    write_csv(results, args.csv_out)
    write_markdown(results, args.baseline, ablations, seeds, args.md_out)
    print(f"[OK] wrote {args.csv_out} and {args.md_out}")


if __name__ == "__main__":
    main()
