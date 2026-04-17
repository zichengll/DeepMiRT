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
