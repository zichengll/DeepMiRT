# Multi-Seed Significance Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce paired multi-seed evidence (n=5) for whether D1_uniform_lr (Δ=+0.26pp) is real and quantify A1_concat (Δ=−6.3pp) significance, against a freshly-run baseline using the same ablation framework.

**Architecture:** Three-phase plan. **Phase A** (one session, ~30 min): create baseline config, write+test analysis script with TDD, smoke-test the launch. **Phase B** (background, ~6.5 days wallclock): execute 13 training runs sequentially via existing `run_ablation.sh`. **Phase C** (one session, ~30 min): run analysis script, write up results. Code state pinned at tag `ablation-pin-2026-04` (commit `4f53f5c` head).

**Tech Stack:** PyTorch Lightning, scipy.stats (paired t-test, Wilcoxon), bash launch scripts, pytest.

**Spec:** `docs/superpowers/specs/2026-04-17-multiseed-significance-design.md`

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `deepmirt/configs/ablation/baseline.yaml` | Create | Minimal ablation framework wrapper around default.yaml so baseline runs land in `checkpoints/ablation/baseline_seed<N>/` parallel to D1/A1 |
| `scripts/analyze_multiseed.py` | Create | Parse `epoch=*-val_auroc=*.ckpt` filenames, run paired t-test + Wilcoxon, emit CSV + markdown |
| `deepmirt/tests/test_analyze_multiseed.py` | Create | Unit tests for parsing and stats wrappers |
| `paper/multi_seed_raw.csv` | Create (Phase C output) | Long format: `config,seed,val_auroc` |
| `paper/multi_seed_significance.md` | Create (Phase C output) | Results table + interpretation |
| `checkpoints/ablation/baseline_seed{42,123,456,789,2024}/` | Create (Phase B output) | New baseline runs |
| `checkpoints/ablation/D1_uniform_lr_seed{123,456,789,2024}/` | Create (Phase B output) | New D1 runs (seed=42 reused) |
| `checkpoints/ablation/A1_concat_seed{123,456,789,2024}/` | Create (Phase B output) | New A1 runs (seed=42 reused) |

---

# Phase A — Pre-launch (one session)

## Task 1: Create baseline.yaml

**Files:**
- Create: `deepmirt/configs/ablation/baseline.yaml`

- [ ] **Step 1: Write the file**

```yaml
# Baseline: default config under the ablation framework.
#
# Purpose: Multi-seed runs at checkpoints/ablation/baseline_seed<N>/
# parallel to D1_uniform_lr_seed<N> and A1_concat_seed<N>, enabling
# paired statistics. Inherits default.yaml unchanged — every ablation
# section field is the framework default.
#
# Spec: docs/superpowers/specs/2026-04-17-multiseed-significance-design.md
_base_: ../default.yaml

ablation:
  interaction: "cross_attention"
  pooling: "mean"
  encoder: "shared"
  random_init: false
  classifier: "mlp"
  uniform_lr: false
```

- [ ] **Step 2: Verify config loads via fast-dev-run**

Run:
```bash
cd /data/home/zicheng/miRNA_target
conda run -n deeplearn python deepmirt/training/train.py \
  --config deepmirt/configs/ablation/baseline.yaml \
  --override seed=999 \
  --override checkpointing.dirpath=/tmp/baseline_smoke_$$  \
  --override logging.log_dir=/tmp/baseline_smoke_logs_$$ \
  --fast-dev-run 2>&1 | tail -30
```

Expected: completes without exception, prints `[INFO] Setting random seed: 999`, hparams.yaml shows `ablation.uniform_lr: false` and `ablation.interaction: cross_attention`. No `.ckpt` files since `--fast-dev-run` skips checkpointing — that's fine.

- [ ] **Step 3: Confirm hparams match D1's baseline assumption**

Compare the loaded hparams of baseline.yaml against D1_uniform_lr.yaml. Both must differ in exactly one field: `uniform_lr`. Run:

```bash
diff <(conda run -n deeplearn python -c "
import yaml, sys
sys.path.insert(0, 'deepmirt/training')
from train import load_config
print(yaml.safe_dump(load_config('deepmirt/configs/ablation/baseline.yaml')['ablation'], sort_keys=True))
") <(conda run -n deeplearn python -c "
import yaml, sys
sys.path.insert(0, 'deepmirt/training')
from train import load_config
print(yaml.safe_dump(load_config('deepmirt/configs/ablation/D1_uniform_lr.yaml')['ablation'], sort_keys=True))
")
```

Expected: only the `uniform_lr` line differs (`false` vs `true`). If `load_config` is not the actual function name in `train.py`, find the equivalent (search for `_deep_merge` and the YAML loader). If anything else differs, fix baseline.yaml so the only difference is `uniform_lr`.

- [ ] **Step 4: Commit**

```bash
git add deepmirt/configs/ablation/baseline.yaml
git commit -m "Add baseline ablation config for multi-seed comparison

Inherits default.yaml unchanged. Enables baseline runs to land in
checkpoints/ablation/baseline_seed<N>/ parallel to D1/A1, so paired
per-seed statistics work without cross-framework drift.

Verified: fast-dev-run loads config; ablation section differs from
D1_uniform_lr.yaml in exactly one field (uniform_lr).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: TDD the val_auroc parser

**Files:**
- Create: `scripts/analyze_multiseed.py` (will grow across tasks 2–4)
- Create: `deepmirt/tests/test_analyze_multiseed.py`

- [ ] **Step 1: Write the failing test**

Create `deepmirt/tests/test_analyze_multiseed.py`:

```python
"""Tests for scripts/analyze_multiseed.py."""
import sys
from pathlib import Path

# scripts/ is not a package — add to sys.path explicitly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import analyze_multiseed as am


def test_best_val_auroc_picks_max_from_filenames(tmp_path):
    (tmp_path / "epoch=12-val_auroc=0.9637.ckpt").touch()
    (tmp_path / "epoch=13-val_auroc=0.9635.ckpt").touch()
    (tmp_path / "epoch=14-val_auroc=0.9610.ckpt").touch()
    (tmp_path / "last.ckpt").touch()  # must be ignored
    assert am.best_val_auroc(tmp_path) == 0.9637


def test_best_val_auroc_raises_on_empty_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        am.best_val_auroc(tmp_path)


def test_best_val_auroc_ignores_last_ckpt(tmp_path):
    (tmp_path / "epoch=01-val_auroc=0.5000.ckpt").touch()
    (tmp_path / "last.ckpt").touch()
    assert am.best_val_auroc(tmp_path) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n deeplearn pytest deepmirt/tests/test_analyze_multiseed.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_multiseed'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/analyze_multiseed.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n deeplearn pytest deepmirt/tests/test_analyze_multiseed.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Sanity check on real data**

```bash
conda run -n deeplearn python -c "
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
from analyze_multiseed import best_val_auroc
print('D1_seed42:', best_val_auroc(Path('checkpoints/ablation/D1_uniform_lr_seed42')))
print('A1_seed42:', best_val_auroc(Path('checkpoints/ablation/A1_concat_seed42')))
"
```

Expected: `D1_seed42: 0.9637`, `A1_seed42: 0.8984` (matches the user's table).

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_multiseed.py deepmirt/tests/test_analyze_multiseed.py
git commit -m "Add val_auroc checkpoint parser for multi-seed analysis

scripts/analyze_multiseed.best_val_auroc() reads top-K ModelCheckpoint
filenames and returns max val_auroc, ignoring last.ckpt. TDD'd with 3
unit tests; sanity-checked against existing D1/A1 seed=42 runs (0.9637
and 0.8984 respectively, matching prior reported numbers).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: TDD the paired stats wrapper

**Files:**
- Modify: `scripts/analyze_multiseed.py`
- Modify: `deepmirt/tests/test_analyze_multiseed.py`

- [ ] **Step 1: Add failing tests**

Append to `deepmirt/tests/test_analyze_multiseed.py`:

```python
import math


def test_paired_stats_a1_like_huge_effect():
    # baseline ~0.96, ablation ~0.90, σ small → p very small
    baseline = [0.961, 0.962, 0.960, 0.963, 0.959]
    ablation = [0.898, 0.899, 0.897, 0.900, 0.896]
    r = am.paired_stats(baseline, ablation)
    assert r["n"] == 5
    assert r["mean_diff"] < -0.05
    assert r["p_ttest"] < 0.001
    assert r["cohens_dz"] < -10  # mean diff dominates std massively
    # 95% CI should not cross zero
    assert r["ci95_high"] < 0


def test_paired_stats_d1_like_borderline():
    # mean diff +0.0026 with non-tiny σ → p around 0.1
    baseline = [0.9611, 0.9605, 0.9618, 0.9608, 0.9614]
    ablation = [0.9637, 0.9620, 0.9645, 0.9628, 0.9640]
    r = am.paired_stats(baseline, ablation)
    assert r["n"] == 5
    assert r["mean_diff"] > 0
    # don't pin exact p — just sanity range
    assert 0 < r["p_ttest"] < 1


def test_paired_stats_zero_diff_is_p_one():
    baseline = [0.95, 0.96, 0.94, 0.97, 0.93]
    ablation = list(baseline)
    r = am.paired_stats(baseline, ablation)
    assert r["mean_diff"] == 0
    assert math.isnan(r["cohens_dz"]) or r["cohens_dz"] == 0
    # p_ttest is 1.0 (or NaN) when all diffs are identically zero
    assert math.isnan(r["p_ttest"]) or r["p_ttest"] >= 0.999


def test_paired_stats_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        am.paired_stats([0.9, 0.9], [0.9, 0.9, 0.9])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n deeplearn pytest deepmirt/tests/test_analyze_multiseed.py -v
```

Expected: 4 new tests fail with `AttributeError: module ... has no attribute 'paired_stats'`. The 3 prior tests still pass.

- [ ] **Step 3: Implement paired_stats**

Append to `scripts/analyze_multiseed.py`:

```python
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
    dz = mean_d / std_d if std_d > 0 else (0.0 if mean_d == 0 else math.nan)

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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
conda run -n deeplearn pytest deepmirt/tests/test_analyze_multiseed.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_multiseed.py deepmirt/tests/test_analyze_multiseed.py
git commit -m "Add paired_stats wrapper with t-test + Wilcoxon + Cohen's dz

scripts/analyze_multiseed.paired_stats() returns mean diff, 95% CI,
paired t-test p, Wilcoxon p, and Cohen's d_z for paired observations.
TDD'd with synthetic A1-like (huge effect, p<<0.001), D1-like
(borderline), zero-diff degenerate, and length-mismatch error cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire up the end-to-end report function

**Files:**
- Modify: `scripts/analyze_multiseed.py`

- [ ] **Step 1: Add CLI + report wiring (no new test — this is glue)**

Append to `scripts/analyze_multiseed.py`:

```python
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
```

- [ ] **Step 2: Smoke-test the CLI on existing partial data**

Only `D1_uniform_lr_seed42` and `A1_concat_seed42` exist; `baseline_*` does not exist yet. The script should warn but not crash. Run:

```bash
conda run -n deeplearn python scripts/analyze_multiseed.py \
  --csv-out /tmp/smoke_raw.csv \
  --md-out /tmp/smoke_sig.md
```

Expected: prints `[WARN] missing: checkpoints/ablation/baseline_seed42` (and 4 more), prints `[OK] wrote ...`. `/tmp/smoke_raw.csv` contains 2 rows (the two seed=42 ablations). `/tmp/smoke_sig.md` per-seed table has `—` everywhere for baseline column. Paired-stats table shows "insufficient data" for both ablations (since baseline has 0 seeds).

- [ ] **Step 3: Re-run all tests**

```bash
conda run -n deeplearn pytest deepmirt/tests/test_analyze_multiseed.py -v
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_multiseed.py
git commit -m "Wire end-to-end CLI for multi-seed analysis

scripts/analyze_multiseed.py main() walks
checkpoints/ablation/<cfg>_seed<S>/, emits paper/multi_seed_raw.csv
(long format) and paper/multi_seed_significance.md (per-seed table +
paired stats table + methods note). Tolerates missing runs with [WARN]
so the script can be invoked mid-launch to inspect partial results.

Smoke-tested on current partial data (only seed=42 D1/A1 exist):
correctly warns on missing baseline and emits 'insufficient data'
rows for the stats table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Smoke-launch verification before committing 6.5 days

**Files:** none (operational)

- [ ] **Step 1: Dry-run all three configs we'll launch**

```bash
cd /data/home/zicheng/miRNA_target
bash scripts/run_ablation.sh --dry-run baseline 2>&1 | tail -10
bash scripts/run_ablation.sh --dry-run D1_uniform_lr 2>&1 | tail -10
bash scripts/run_ablation.sh --dry-run A1_concat 2>&1 | tail -10
```

Expected: each prints `[CMD] python ... --fast-dev-run` then completes with `[DONE]`. No tracebacks. (Note: `--dry-run` adds `--fast-dev-run` per the script — runs ~1 epoch on tiny batches.)

If `[SKIP]` triggers because `<exp>_seed42` already exists, that's expected for D1 and A1 — they have prior runs. Baseline should NOT skip.

- [ ] **Step 2: Verify GPUs free**

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Expected: both L20s near zero memory and 0% util. If not, identify what's running before launching.

- [ ] **Step 3: Verify free disk for ~13 × ~1.5GB = ~20GB of new checkpoints**

```bash
df -h /data/home/zicheng/miRNA_target/checkpoints/
du -sh /data/home/zicheng/miRNA_target/checkpoints/ablation/A1_concat_seed42/
```

Expected: `df` shows >50GB free. The `du` gives a per-run reference — multiply by 13 for budget.

---

## Task 6: Launch the 13 background runs

**Files:** none (operational; produces `checkpoints/ablation/...` dirs)

- [ ] **Step 1: Launch baseline (5 seeds, sequential)**

```bash
cd /data/home/zicheng/miRNA_target
nohup bash scripts/run_ablation.sh baseline \
  --seeds 42,123,456,789,2024 \
  > checkpoints/ablation/baseline_multiseed.log 2>&1 &
echo "baseline launcher PID: $!"
```

Expected wallclock: ~2.5 days (5 × ~12h sequential). Records PID; the launcher script itself runs sequentially, blocking on each `python train.py`.

- [ ] **Step 2: Wait for baseline to finish, then launch D1**

DO NOT launch D1/A1 in parallel — they share the same 2 GPUs as baseline (DDP, devices=auto). Watch with:

```bash
tail -f /data/home/zicheng/miRNA_target/checkpoints/ablation/baseline_multiseed.log
```

When it prints `All ablation experiments completed!`, launch D1:

```bash
nohup bash scripts/run_ablation.sh D1_uniform_lr \
  --seeds 123,456,789,2024 \
  > checkpoints/ablation/D1_multiseed.log 2>&1 &
echo "D1 launcher PID: $!"
```

(seed=42 is reused — `[SKIP]` will fire if you accidentally include it.)

Expected wallclock: ~2 days (4 × ~12h).

- [ ] **Step 3: After D1 finishes, launch A1**

```bash
nohup bash scripts/run_ablation.sh A1_concat \
  --seeds 123,456,789,2024 \
  > checkpoints/ablation/A1_multiseed.log 2>&1 &
echo "A1 launcher PID: $!"
```

Expected wallclock: ~2 days.

- [ ] **Step 4: Sanity-check progress at any time**

```bash
ls /data/home/zicheng/miRNA_target/checkpoints/ablation/baseline_seed* /data/home/zicheng/miRNA_target/checkpoints/ablation/D1_uniform_lr_seed* /data/home/zicheng/miRNA_target/checkpoints/ablation/A1_concat_seed* 2>/dev/null
```

A finished run has 3 `epoch=*-val_auroc=*.ckpt` files + 1 `last.ckpt`. A failed/aborted run is detectable by missing `epoch=*` files. Re-launch a single failed pair with `FORCE=1 bash scripts/run_ablation.sh <exp> --seeds <S>` after manually clearing the dir.

---

# Phase C — Post-launch analysis (one session, days later)

## Task 7: Run analysis and commit results

**Files:**
- Create: `paper/multi_seed_raw.csv`
- Create: `paper/multi_seed_significance.md`

- [ ] **Step 1: Verify all 13 new runs completed**

```bash
cd /data/home/zicheng/miRNA_target
for cfg in baseline D1_uniform_lr A1_concat; do
  for s in 42 123 456 789 2024; do
    d="checkpoints/ablation/${cfg}_seed${s}"
    n=$(ls "$d"/epoch=*.ckpt 2>/dev/null | wc -l)
    echo "$d: $n ckpts"
  done
done
```

Expected: every row shows `3 ckpts` (D1_seed42 and A1_seed42 are reused; baseline_seed42 was newly run). Any row with 0 ckpts means a failed run — re-launch that single pair before proceeding (see Task 6 Step 4).

- [ ] **Step 2: Run analysis**

```bash
conda run -n deeplearn python scripts/analyze_multiseed.py
```

Expected: prints `[OK] wrote paper/multi_seed_raw.csv and paper/multi_seed_significance.md`. No `[WARN]` lines if all runs completed.

- [ ] **Step 3: Eyeball the markdown**

```bash
cat paper/multi_seed_significance.md
```

Sanity checks:
- A1 row: `mean Δ` near −6.0 to −6.5 pp, `paired t p` < 0.001, `Cohen's d_z` < −5, `sig` = ✅
- D1 row: `mean Δ` somewhere near +0.3 pp ± a lot, `paired t p` is the actual answer to the question — could be 0.001 or 0.5 depending on variance
- Per-seed table fully populated, no `—`

If any number looks impossible (e.g. p > 1, NaN where it shouldn't be), inspect `paper/multi_seed_raw.csv` and re-derive by hand for that ablation.

- [ ] **Step 4: Commit**

```bash
git add paper/multi_seed_raw.csv paper/multi_seed_significance.md
git commit -m "Add multi-seed significance results for D1 and A1 ablations

5-seed paired comparison against baseline. Spec at
docs/superpowers/specs/2026-04-17-multiseed-significance-design.md.
Code pin: tag ablation-pin-2026-04. Raw CSV preserves per-seed
val_auroc for re-analysis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Hand off to user for paper writeup**

The markdown is intended as the data source for the paper's "ablation significance" paragraph. Do not auto-edit `paper/main.tex` — leave the wording to the author so the framing matches the paper voice.

---

## Self-Review (writer checklist)

- **Spec coverage:** Every deliverable in spec §Deliverables maps to a task — baseline.yaml (T1), analyze script (T2-4), raw CSV + md (T7). Methods note about DDP/seed sources is in T4 markdown template (lines 191-198). Skip-if-exists already shipped pre-plan.
- **Placeholder scan:** No TBD/TODO. Each step has either a code block or an exact command.
- **Type/name consistency:** `best_val_auroc(run_dir: Path)` → used in T2/T4 with consistent signature. `paired_stats(baseline, ablation)` returns dict with keys used in T4 markdown writer (`mean_diff`, `ci95_low`, `ci95_high`, `p_ttest`, `p_wilcoxon`, `cohens_dz`, `n`) — all defined in T3 implementation.
- **Order:** Phase A is reversible/cheap; Phase B is the 6.5-day commit; Phase C requires Phase B output. Smoke-test (T5) sits before launch (T6) so a misconfig is caught before burning compute.
