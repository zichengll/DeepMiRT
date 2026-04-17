# Multi-Seed Significance Verification for D1 (uniform_lr) and A1 (concat) Ablations

**Date:** 2026-04-17
**Author:** zicheng (with Claude Code)
**Status:** Approved, ready for implementation plan

## Background

Single-seed (seed=42) ablation results showed:

- **D1_uniform_lr**: val_auroc 0.9637, Δ = +0.26pp vs baseline (0.9611) — only ablation that beats baseline; suspected noise.
- **A1_concat**: val_auroc 0.8984, Δ = −6.3pp vs baseline — largest degradation in the table; needs a number for the paper.

Single-seed point estimates are insufficient to claim either result is reproducible. The +0.26pp gap on D1 in particular is plausibly within seed-to-seed variance.

The "baseline 0.9611" reference comes from the main training run, not from a run inside the ablation framework, so paired statistics against ablation runs are not yet possible.

## Goal

Produce paired multi-seed evidence for two claims:

1. **D1**: Whether uniform learning rate is genuinely better than the layerwise default, or noise.
2. **A1**: Whether replacing cross-attention with concat is significantly worse (expected, but quantified).

## Scope

### In scope

- Run **baseline** (default config) under the ablation framework with 5 seeds.
- Run **D1_uniform_lr** with 4 additional seeds (reuse existing seed=42).
- Run **A1_concat** with 4 additional seeds (reuse existing seed=42).
- Statistical analysis: paired t-test (primary) + Wilcoxon signed-rank (secondary).
- Markdown writeup + raw CSV + reusable analysis script.

### Out of scope

- Multi-seed for other ablation cells (C2_max_pooling etc.) — explicitly deferred.
- Hyperparameter retuning of D1.
- External benchmark (mirBench) re-evaluation.

## Experiment Matrix

| Config | seed=42 | seeds to run | Source config |
|---|---|---|---|
| baseline | (does not exist in ablation/) | 42, 123, 456, 789, 2024 | new `deepmirt/configs/ablation/baseline.yaml` (only `_base_: ../default.yaml`) |
| D1_uniform_lr | 0.9637 (reuse) | 123, 456, 789, 2024 | existing |
| A1_concat | 0.8984 (reuse) | 123, 456, 789, 2024 | existing |

**Total new training runs:** 13.

## Compute Plan

- **Per-run setup:** unchanged — 2× L20 with DDP (`devices: auto`, `strategy: ddp`), batch_size=512, max_epochs=20.
- **Why not split GPUs across runs:** halving DDP world size changes effective batch size. Comparing baseline (DDP) vs D1/A1 (single-GPU) would introduce a confounder. Sequential is cleaner.
- **Estimated wallclock:** ~12 h/run × 13 runs ≈ 6.5 days, sequential.
- **Launch:** existing `scripts/run_ablation.sh` with `--seeds 123,456,789,2024` (and one separate invocation with `--seeds 42,123,456,789,2024` for baseline since it has no prior run). Run in background via `nohup`.

## Result Extraction

Each run already saves `epoch=<E>-val_auroc=<X>.ckpt` files (top-3 by val_auroc, monitor=val_auroc, mode=max). Take the maximum val_auroc parsed from filenames in each run's checkpoint directory — no re-evaluation needed, no TensorBoard parsing needed.

Pseudocode:
```python
def best_val_auroc(run_dir: Path) -> float:
    return max(
        float(re.search(r"val_auroc=([0-9.]+)", p.name).group(1))
        for p in run_dir.glob("epoch=*-val_auroc=*.ckpt")
    )
```

## Statistical Analysis

For each ablation X ∈ {D1, A1}, with paired observations (baseline_seed_i, X_seed_i) for i ∈ {42, 123, 456, 789, 2024}:

- **Primary:** paired t-test on differences `d_i = X_seed_i − baseline_seed_i`. Report mean d, std d, 95% CI (t-distribution, df=4), p-value, Cohen's d_z = mean(d) / std(d).
- **Secondary:** Wilcoxon signed-rank on the same pairs (low power at n=5 but non-parametric sanity check).
- **Multiple-comparisons correction:** Bonferroni for 2 tests → α = 0.025.

Implementation: `scipy.stats.ttest_rel`, `scipy.stats.wilcoxon`, `scipy.stats.t.interval`.

### Expected outcomes

- **A1**: ~6pp gap dominates any plausible seed variance → paired t p ≪ 0.001. Result is essentially a number to cite.
- **D1**: This is the real question. If per-seed std of (D1 − baseline) is ≈ 0.3pp, p ≈ 0.1 (not significant). If std ≈ 0.1pp, p < 0.05. Either outcome is publishable: "the only ablation cell beating baseline is/is not statistically significant."

## Deliverables

1. `deepmirt/configs/ablation/baseline.yaml` — minimal config inheriting default.yaml.
2. `scripts/analyze_multiseed.py` — extract per-run best val_auroc, run paired tests, write CSV + markdown.
3. `paper/multi_seed_raw.csv` — long-format: `config, seed, val_auroc`.
4. `paper/multi_seed_significance.md` — table (mean ± std, 95% CI, p, Cohen's d) + brief interpretation paragraph.
5. New checkpoint directories under `checkpoints/ablation/{baseline,D1_uniform_lr,A1_concat}_seed<S>/` for the 13 new runs.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 6.5-day wallclock; user may need GPUs | Launch in background; user can interrupt/resume per-experiment via the existing script |
| A run crashes mid-way | `set -euo pipefail` aborts the script; user re-runs only the failed config/seed (script is per-run idempotent at the directory level — re-running overwrites checkpoints in that seed dir, which is acceptable) |
| Data shuffling not seeded → between-seed variance underestimates true variance | Verify `seed=N` propagates to data shuffling (Lightning's `seed_everything` is called in `train.py`); if not, file as separate bug |
| n=5 still under-powered for D1 | Accept and report as such. The paper writeup distinguishes "evidence of effect" (p<0.025) from "no evidence" (p>0.025), not "evidence of no effect" |

## Non-Decisions (Explicitly Deferred)

- Whether to retune D1 hyperparameters if it does prove significant.
- Whether to extend to all 16 ablation cells with multi-seed (the user explicitly chose minimal set).
- CLASH-task gap with miRBind (already noted in README as known limitation).
