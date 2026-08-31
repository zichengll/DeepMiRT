# Ablation Results

Source files:

- `paper/multi_seed_raw.csv`
- `paper/multi_seed_significance.md`

## Summary

All ablations were evaluated with five paired random seeds: 42, 123, 456, 789, and 2024. The reported metric is validation AUROC. The baseline uses the shared RNA-FM encoder, two-layer cross-attention interaction module, masked mean pooling, MLP classifier, and differential learning rates. `A1_concat` removes cross-attention by mean-pooling miRNA and target representations separately and concatenating them before classification. `D1_uniform_lr` keeps the baseline architecture unchanged but uses the same base learning rate for all optimized parameter groups.

| Model | Validation AUROC, mean over 5 seeds | Mean delta vs baseline (pp) | 95% CI for delta (pp) | Paired t-test p |
|---|---:|---:|---:|---:|
| Baseline | 0.9509 | 0.000 | - | - |
| D1_uniform_lr | 0.9637 | +1.278 | [+1.234, +1.322] | 1.45e-07 |
| A1_concat | 0.8983 | -5.258 | [-5.304, -5.212] | 5.893e-10 |

## Per-seed Results

| Seed | Baseline | D1_uniform_lr | A1_concat |
|---:|---:|---:|---:|
| 42 | 0.9508 | 0.9637 | 0.8984 |
| 123 | 0.9507 | 0.9638 | 0.8982 |
| 456 | 0.9511 | 0.9635 | 0.8981 |
| 789 | 0.9513 | 0.9637 | 0.8984 |
| 2024 | 0.9506 | 0.9637 | 0.8985 |

## Interpretation

The concat ablation shows that cross-attention is a major contributor to model performance. Replacing cross-attention with independent mean pooling and concatenation reduces validation AUROC from 0.9509 to 0.8983, a mean drop of 5.258 percentage points across paired seeds. The drop is consistent across all five seeds and is significant under the paired t-test after Bonferroni correction. This supports the architectural claim that explicitly modeling miRNA-target interaction through cross-attention is more effective than passing independently pooled sequence embeddings to the classifier.

The uniform-learning-rate ablation improves validation AUROC from 0.9509 to 0.9637, a mean gain of 1.278 percentage points. Because the architecture is unchanged, this result points to optimizer configuration rather than representational capacity. In the current frozen-backbone setting, the baseline differential learning-rate schedule is likely too conservative for the cross-attention module, while using the same base learning rate across optimized modules allows the interaction layer and classifier to adapt more effectively.

## Paper-ready Paragraph

We performed paired multi-seed ablations to quantify the contribution of cross-attention and the optimizer schedule. Across five seeds, replacing cross-attention with independent mean pooling followed by concatenation reduced validation AUROC from 0.9509 to 0.8983, corresponding to a mean decrease of 5.258 percentage points (95% CI: -5.304 to -5.212; paired t-test p = 5.893e-10). This confirms that explicit interaction modeling through cross-attention is central to DeepMiRT's predictive performance. In contrast, keeping the architecture fixed but using a uniform learning rate improved validation AUROC to 0.9637, a mean gain of 1.278 percentage points over the baseline (95% CI: +1.234 to +1.322; paired t-test p = 1.45e-07). Thus, while the cross-attention architecture is strongly supported, the differential learning-rate schedule is not optimal in the current frozen-backbone training setup.
