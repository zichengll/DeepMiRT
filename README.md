<p align="center">
  <h1 align="center">DeepMiRT</h1>
  <p align="center">
    <strong>miRNA Target Prediction with RNA Foundation Models</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/deepmirt/"><img src="https://img.shields.io/pypi/v/deepmirt?color=blue" alt="PyPI"></a>
    <a href="https://huggingface.co/liuliu2333/deepmirt"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model-orange" alt="HF Model"></a>
    <a href="https://huggingface.co/spaces/liuliu2333/deepmirt"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Demo-yellow" alt="HF Demo"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"></a>
    <img src="https://img.shields.io/badge/internal_AUROC-0.96-brightgreen" alt="Internal AUROC 0.96">
    <img src="https://img.shields.io/badge/eCLIP_AUROC-0.75-blue" alt="eCLIP AUROC 0.75">
  </p>
</p>

DeepMiRT predicts miRNA-target interactions using [RNA-FM](https://github.com/ml4bio/RNA-FM) embeddings and cross-attention. It ranks **#1 on eCLIP benchmarks** among 12 methods in both AUROC (~0.75) and APS (average precision), and reaches **0.96 AUROC** on a held-out test set of 813K samples.

---

## Why DeepMiRT?

Existing miRNA target prediction tools rely on hand-crafted thermodynamic rules or shallow sequence features, struggling to capture the full complexity of miRNA-target recognition. DeepMiRT addresses this by leveraging **RNA-FM**, a foundation model pre-trained on 23 million non-coding RNAs, as a shared encoder for both miRNA and target. A **cross-attention** mechanism then lets the target "read" the miRNA to learn complementarity patterns beyond simple seed matching. The result: state-of-the-art performance, ranking **#1 among 12 methods** on eCLIP benchmarks (AUROC 0.7511 / 0.7524; APS up to 0.7947) and achieving **0.96 AUROC** on a held-out test set of 813K samples.

## Key Results at a Glance

<table>
<tr>
<td align="center"><strong>0.96</strong><br>Internal AUROC<br>(813K test set)</td>
<td align="center"><strong>#1 / 12</strong><br>eCLIP Benchmark<br>(AUROC ~0.75)</td>
<td align="center"><strong>0.7947</strong><br>Best eCLIP APS<br>(Manakov)</td>
<td align="center"><strong>3-line</strong><br>Python API</td>
</tr>
</table>

## Quick Start

```bash
pip install deepmirt
```

```python
from deepmirt import predict

probs = predict(
    mirna_seqs=["UGAGGUAGUAGGUUGUAUAGUU"],
    target_seqs=["ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU"],
)
print(f"Interaction probability: {probs[0]:.4f}")
```

Model weights are automatically downloaded from Hugging Face Hub on first use (~495 MB).

## Architecture

```mermaid
graph LR
    A["miRNA<br/>(18-25 nt)"] --> B["RNA-FM<br/>Encoder"]
    C["Target<br/>(40 nt)"] --> D["RNA-FM<br/>Encoder"]
    B --> E["miRNA Embedding<br/>(B, M, 640)"]
    D --> F["Target Embedding<br/>(B, T, 640)"]
    E --> G["Cross-Attention<br/>(2 layers, 8 heads)"]
    F --> G
    G --> H["Masked Mean Pool"]
    H --> I["MLP Head<br/>640 → 256 → 64 → 1"]
    I --> J["Probability"]

    style B fill:#4a90d9,color:#fff
    style D fill:#4a90d9,color:#fff
    style G fill:#e67e22,color:#fff
    style I fill:#27ae60,color:#fff
```

The model uses a **shared RNA-FM encoder** for both sequences, ensuring they lie in the same representation space. Cross-attention lets the target "read" the miRNA to capture complementarity and binding signals. The RNA-FM backbone is kept frozen during training; only the cross-attention and classifier head are optimized (~4M trainable parameters out of 103M total).

<details>
<summary>ASCII diagram (fallback)</summary>

```
miRNA  (18-25 nt) ──→ [RNA-FM Encoder] ──→ miRNA embedding (B, M, 640) ──────────┐
                        (shared weights)                                           ↓
Target (40 nt)    ──→ [RNA-FM Encoder] ──→ target embedding (B, T, 640) → [Cross-Attention]
                                                                                   ↓
                                                                          Masked Mean Pool
                                                                                   ↓
                                                                              [MLP Head]
                                                                           640 → 256 → 64 → 1
                                                                                   ↓
                                                                             probability
```

</details>

## Genome-Wide Target Scanning

DeepMiRT can scan entire 3'UTR or transcript sequences to identify binding sites genome-wide -- similar to miRanda, but powered by the deep learning model.

### Python API

```python
from deepmirt import scan_targets

results = scan_targets(
    mirna_fasta="mirnas.fa",           # or dict: {"let-7": "UGAGGUAGUAGGUUGUAUAGUU"}
    target_fasta="3utrs.fa",
    output_prefix="results/scan",      # writes _details.txt, _hits.tsv, _summary.tsv
    device="cuda",
    scan_mode="hybrid",                # "seed" | "hybrid" | "exhaustive"
    prob_threshold=0.5,
)

for r in results:
    for hit in r.hits:
        print(f"{r.target_id} pos={hit.position} prob={hit.probability:.3f} ({hit.seed_type})")
```

### Command Line

```bash
# Scan with a FASTA of miRNAs against target 3'UTRs
deepmirt-predict scan \
    --mirna-fasta mirnas.fa \
    --target-fasta 3utrs.fa \
    --output results/scan \
    --device cuda \
    --scan-mode hybrid \
    --threshold 0.5

# Scan with a single miRNA sequence
deepmirt-predict scan \
    --mirna UGAGGUAGUAGGUUGUAUAGUU \
    --mirna-id dme-let-7 \
    --target-fasta 3utrs.fa \
    --output results/scan \
    --device cpu
```

### Scanning Modes

| Mode | Strategy | Speed | Use Case |
|------|----------|-------|----------|
| `seed` | Only score seed match positions (8mer/7mer/6mer) | Fastest | Quick screen, high-confidence sites |
| `hybrid` | Seed matches + sliding window (stride=20) to fill gaps | Default | Balanced: catches seed + non-canonical sites |
| `exhaustive` | Sliding window across entire target | Slowest | Comprehensive, catches everything |

### Output Files

| File | Description |
|------|-------------|
| `{prefix}_details.txt` | Human-readable report with ASCII alignment for each hit |
| `{prefix}_hits.tsv` | Per-hit table: position, probability, seed type, window sequence |
| `{prefix}_summary.tsv` | Per-target summary: number of hits, max/mean probability |

Example alignment from `_details.txt`:

```
Scanning: dme-miR-1-3p vs FBTR0082186_3UTR (1247 nt)
  Hits found: 2

  Hit at position 617, Prob: 0.8923, Seed: 7mer-m8

    miRNA  3' ...uauCCGCGGCCggg... 5'
                  ||||||||::
    Target 5' ...aGGCGCCGGAact... 3'
```

## Installation

### From PyPI

```bash
pip install deepmirt
```

### From source (for development)

```bash
git clone https://github.com/zichengll/DeepMiRT.git
cd DeepMiRT
pip install -e ".[dev,eval]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 1.12
- RNA-FM (`rna-fm` package)
- ~495 MB disk space for model weights (auto-downloaded)

## Usage

### Python API

```python
from deepmirt import predict

# Single pair
probs = predict(
    mirna_seqs=["UGAGGUAGUAGGUUGUAUAGUU"],
    target_seqs=["ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU"],
)

# Multiple pairs
probs = predict(
    mirna_seqs=["UGAGGUAGUAGGUUGUAUAGUU", "UAGCAGCACGUAAAUAUUGGCG"],
    target_seqs=["ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU",
                 "GCAAUGUUUUCCACAGUGCUUACACAGAAAUAGCAACUUUA"],
    device="cuda",  # use GPU if available
)
```

### CSV Batch Prediction

```python
from deepmirt.predict import predict_from_csv

df = predict_from_csv(
    csv_path="input.csv",         # must have mirna_seq and target_seq columns
    output_path="results.csv",
    device="cpu",
)
```

### Command Line

```bash
# Single pair
deepmirt-predict single --mirna UGAGGUAGUAGGUUGUAUAGUU \
    --target ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU

# Batch from CSV
deepmirt-predict batch --input data.csv --output results.csv --device cuda
```

### Input Format

- **miRNA sequences**: 18-25 nt, DNA (T) or RNA (U) format
- **Target sequences**: 40 nt recommended (the model was trained on 40-nt target site fragments)
- Sequences are automatically converted to RNA format internally

### Web Demo

Try DeepMiRT without installation:
**[huggingface.co/spaces/liuliu2333/deepmirt](https://huggingface.co/spaces/liuliu2333/deepmirt)**

The demo supports single-pair prediction with pre-loaded examples and batch CSV upload.

## Benchmark Results

### Standard Benchmark: miRBench eCLIP Datasets

DeepMiRT ranks **#1** on both eCLIP benchmark datasets from miRBench in both APS (average precision, miRBench's primary metric) and AUROC. All methods evaluated on identical held-out data using the miRBench framework:

| Method | Type | Klimentova APS | Klimentova AUROC | Manakov APS | Manakov AUROC |
|--------|------|---------------|-----------------|------------|--------------|
| **DeepMiRT (ours)** | **DL + LM** | **0.7850** | **0.7511** | **0.7947** | **0.7524** |
| TargetScanCnn | CNN | 0.7447 | 0.7138 | 0.7735 | 0.7222 |
| miRBind | DL | 0.7530 | 0.7004 | 0.7090 | 0.6728 |
| miRNA_CNN | CNN | 0.7392 | 0.6981 | 0.7113 | 0.6842 |
| RNACofold | Thermo. | 0.6685 | 0.6740 | 0.6280 | 0.6299 |

### Standard Benchmark: miRBench CLASH Dataset

On the CLASH dataset, DeepMiRT ranks #5 (honest reporting -- CLASH captures different biology):

| Method | APS | AUROC |
|--------|-----|-------|
| miRBind | 0.7964 | 0.7649 |
| miRNA_CNN | 0.7726 | 0.7614 |
| InteractionAwareModel | 0.7414 | 0.7510 |
| RNACofold | 0.7394 | 0.7455 |
| **DeepMiRT (ours)** | **0.7260** | **0.6952** |

### Our Test Set (813K samples)

| Method | Type | AUROC | AUPRC | F1 | MCC |
|--------|------|-------|-------|----|-----|
| **DeepMiRT (ours)** | **DL + LM** | **0.9606** | **0.9669** | **0.8949** | **0.8111** |
| TargetScanCnn | CNN | 0.8856 | 0.8999 | 0.8449 | 0.7197 |
| Seed Match | Rule | 0.8817 | 0.8585 | 0.8719 | 0.7726 |
| miRanda | Complement + MFE | 0.7688 | 0.7168 | 0.7813 | 0.5858 |
| miRBind | DL | 0.7641 | 0.7446 | 0.7012 | 0.3723 |
| RNAhybrid | MFE | 0.7230 | 0.7079 | 0.6673 | 0.3406 |

<details>
<summary>Full comparison table (16 methods)</summary>

| Method | Type | AUROC | AUPRC | F1 | MCC | Sensitivity | Specificity |
|--------|------|-------|-------|----|-----|-------------|-------------|
| DeepMiRT (ours) | DL + LM | 0.9606 | 0.9669 | 0.8949 | 0.8111 | 0.8396 | 0.9641 |
| TargetScanCnn | CNN | 0.8856 | 0.8999 | 0.8449 | 0.7197 | 0.7944 | 0.9182 |
| Seed Match | Rule | 0.8817 | 0.8585 | 0.8719 | 0.7726 | 0.8098 | 0.9535 |
| Seed6mer | Rule | 0.8670 | 0.8241 | 0.8592 | 0.7374 | 0.8263 | 0.9077 |
| Seed7mer | Rule | 0.7802 | 0.7619 | 0.7268 | 0.6118 | 0.5865 | 0.9740 |
| miRanda | Complement + MFE | 0.7688 | 0.7168 | 0.7813 | 0.5858 | 0.7420 | 0.8410 |
| miRBind | DL | 0.7641 | 0.7446 | 0.7012 | 0.3723 | 0.7659 | 0.6021 |
| miRNA_CNN | CNN | 0.7299 | 0.7200 | 0.6555 | 0.3543 | 0.6296 | 0.7229 |
| RNAhybrid | MFE | 0.7230 | 0.7079 | 0.6673 | 0.3406 | 0.6582 | 0.6823 |
| Seed6merBulgeOrMismatch | Rule | 0.6974 | 0.6142 | 0.7423 | 0.4339 | 0.9101 | 0.4846 |
| RNACofold | Thermo. | 0.6618 | 0.6262 | 0.6192 | 0.2417 | 0.6329 | 0.6088 |
| InteractionAwareModel | DL + Attn | 0.6473 | 0.6145 | 0.6008 | 0.2189 | 0.6026 | 0.6164 |
| Seed8mer | Rule | 0.5656 | 0.5519 | 0.2394 | 0.2581 | 0.1367 | 0.9945 |
| Random | -- | 0.4998 | 0.4928 | 0.4880 | 0.0009 | 0.4826 | 0.5182 |
| TargetNet | DL | 0.4743 | 0.4715 | 0.0214 | 0.0076 | 0.0109 | 0.9906 |
| CnnMirTarget | CNN | 0.4260 | 0.4414 | 0.0029 | 0.0048 | 0.0015 | 0.9989 |

</details>

![Benchmark Comparison](docs/benchmark_comparison.png)

> **Note:** DeepMiRT excels at target site identification (eCLIP-type tasks) but is less suited for distinguishing competitive binding among multiple miRNAs at the same target site (CLASH-type tasks).

<details>
<summary>Training your own model</summary>

## Training

```bash
# Train with frozen backbone (recommended — this is how the released model was trained)
python deepmirt/training/train.py \
    --config deepmirt/configs/default.yaml
```

The config also supports experimental progressive unfreezing of the top RNA-FM layers (`unfreezing.enabled: true`), but in our experiments this did not improve performance over the frozen-backbone baseline.

See `deepmirt/configs/default.yaml` for all configurable hyperparameters.

</details>

<details>
<summary>Project structure</summary>

## Project Structure

```
DeepMiRT/
├── deepmirt/
│   ├── model/              # Neural network modules
│   │   ├── mirna_target_model.py   # Full model: RNA-FM + CrossAttn + MLP
│   │   ├── rnafm_encoder.py        # RNA-FM wrapper with freeze/unfreeze
│   │   ├── cross_attention.py      # Multi-layer cross-attention block
│   │   └── classifier.py           # MLP classification head
│   ├── training/           # PyTorch Lightning training
│   │   ├── lightning_module.py     # LightningModule with metrics
│   │   ├── train.py               # Training entry point
│   │   └── callbacks.py           # Staged unfreezing callback
│   ├── data_module/        # Data loading and preprocessing
│   ├── evaluation/         # 9-step evaluation pipeline
│   ├── scanning/           # Genome-wide target site scanning
│   │   ├── scanner.py             # Core TargetScanner class
│   │   ├── site_finder.py         # Seed match finder
│   │   └── output_formatter.py    # TXT/TSV output formatters
│   ├── configs/            # YAML configuration files
│   ├── predict.py          # Public prediction & scanning API
│   └── tests/              # Unit tests
├── app.py                  # Gradio web demo
├── examples/               # Usage examples
└── docs/                   # Figures and documentation
```

</details>

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest deepmirt/tests/`
5. Open a pull request

## Citation

```bibtex
@software{liu2026deepmirt,
  title={DeepMiRT: miRNA Target Prediction with RNA Foundation Models},
  author={Liu, Zicheng},
  year={2026},
  url={https://github.com/zichengll/DeepMiRT}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [RNA-FM](https://github.com/ml4bio/RNA-FM) -- pre-trained RNA foundation model (Chen et al., 2022)
- [miRBench](https://github.com/katarinagresova/miRBench) -- standardized benchmark framework (Gresova et al.)
- [PyTorch Lightning](https://lightning.ai/) -- training framework
- [Hugging Face](https://huggingface.co/) -- model hosting and demo platform
