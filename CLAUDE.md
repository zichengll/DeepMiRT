# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepMiRT: miRNA target prediction using RNA-FM (pre-trained 12-layer Transformer) + cross-attention + MLP classifier. Binary classification of miRNA-target interactions. Built with PyTorch Lightning, trained on 2x NVIDIA L20 GPUs with DDP.

## Common Commands

### Linting
```bash
ruff check deepmirt/
```

### Testing
```bash
# Unit tests (no GPU needed, used in CI)
pytest deepmirt/tests/test_unit.py -v

# Smoke tests (requires GPU and RNA-FM model)
conda run -n deeplearn python -m pytest deepmirt/tests/test_smoke.py -v --tb=short
```

### Training
```bash
# Smoke test (quick sanity check)
python deepmirt/training/train.py --config deepmirt/configs/default.yaml --fast-dev-run

# Full training (Phase 1: frozen backbone)
python deepmirt/training/train.py --config deepmirt/configs/default.yaml

# Override parameters
python deepmirt/training/train.py --config deepmirt/configs/default.yaml \
  --override training.lr=5e-5 --override data.batch_size=64

# Phase 2: unfreeze top layers from checkpoint
python deepmirt/training/train.py --config deepmirt/configs/default.yaml \
  --ckpt-path checkpoints/best.ckpt --override unfreezing.enabled=true
```

### Ablation Studies
```bash
bash scripts/run_ablation.sh --dry-run              # Smoke test
bash scripts/run_ablation.sh A1_concat               # Single experiment
bash scripts/run_ablation.sh --p0                    # Priority 0 group (5 experiments)
bash scripts/run_ablation.sh --all --seeds 42,123    # All experiments, multiple seeds
```

### Install (dev)
```bash
pip install -e ".[dev,eval]"
```

## Architecture

**Data flow:** miRNA sequence + target sequence (40 nt) -> shared RNA-FM encoder -> cross-attention (target queries miRNA, 2 layers, 8 heads) -> masked mean pooling -> MLP head (640->256->64->1) -> probability.

Key model files in `deepmirt/model/`:
- `mirna_target_model.py` — main model assembling encoder + cross-attention + classifier; contains ablation conditional branches
- `rnafm_encoder.py` — RNA-FM wrapper with freeze/unfreeze support
- `cross_attention.py` — multi-layer cross-attention block
- `classifier.py` — MLP classification head (supports linear-only ablation)

Training files in `deepmirt/training/`:
- `train.py` — entry point; YAML config loading with `_base_` inheritance via `_deep_merge`, DDP/NCCL setup
- `lightning_module.py` — LightningModule with train/val/test steps, metrics, optimizer config
- `callbacks.py` — staged unfreezing callback

## Config System

- Main config: `deepmirt/configs/default.yaml`
- Ablation configs in `deepmirt/configs/ablation/` use `_base_: ../default.yaml` inheritance
- The `ablation` section in config has 6 dimensions: `interaction`, `pooling`, `encoder`, `random_init`, `classifier`, `uniform_lr`
- Model code reads ablation params via conditional branches (no separate model files per ablation)

## Code Style

- Ruff: line-length 100, selects E/F/I/W, ignores E501
- `train.py` has E402 exception (delayed imports after NCCL env setup)
- Python >= 3.9, target 3.10 for CI

## CI

GitHub Actions (`.github/workflows/ci.yaml`): runs `ruff check` and `pytest deepmirt/tests/test_unit.py` on push/PR to main.

## Environment

- Conda environment: `deeplearn`
- CLI entry point: `deepmirt-predict` (defined in pyproject.toml)
- Model weights (~495 MB) auto-download from Hugging Face Hub on first use
