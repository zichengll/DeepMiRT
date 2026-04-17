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
