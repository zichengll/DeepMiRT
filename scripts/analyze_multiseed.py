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
