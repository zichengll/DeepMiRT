# Multi-Seed Significance: D1 (uniform_lr) and A1 (concat)

**Baseline:** `baseline` (default.yaml under ablation framework)  
**Seeds requested:** [42, 123, 456, 789, 2024]  
**Significance threshold:** α = 0.025 (Bonferroni-corrected for 2 tests)  
**Code pin:** git tag `ablation-pin-2026-04`

## Per-seed val_auroc

| seed | baseline | D1_uniform_lr | A1_concat |
|---|---|---|---|
| 42 | 0.9508 | 0.9637 | 0.8984 |
| 123 | 0.9507 | 0.9638 | 0.8982 |
| 456 | 0.9511 | 0.9635 | 0.8981 |
| 789 | 0.9513 | 0.9637 | 0.8984 |
| 2024 | 0.9506 | 0.9637 | 0.8985 |

## Paired statistics vs baseline

| ablation | n | mean Δ (pp) | 95% CI (pp) | paired t p | Wilcoxon p | Cohen's d_z | sig at α=0.025 |
|---|---|---|---|---|---|---|---|
| D1_uniform_lr | 5 | +1.278 | [+1.234, +1.322] | 1.45e-07 | 0.0625 | +35.86 | ✅ |
| A1_concat | 5 | -5.258 | [-5.304, -5.212] | 5.893e-10 | 0.0625 | -142.06 | ✅ |

## Methods note

Each seed varies three random sources jointly: data shuffling order, model weight initialization, and DDP allreduce ordering under bf16. We do not enable `torch.use_deterministic_algorithms(True)` because the ~20% throughput cost is not justified at this experiment scale; the remaining non-determinism is folded into the seed-to-seed variance estimate. n=5 is at the lower end of paired ablation reporting; for borderline effects we report `p_wilcoxon` as a non-parametric sanity check but do not include it in the Bonferroni family.
