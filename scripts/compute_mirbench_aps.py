#!/usr/bin/env python3
"""
Compute APS (Average Precision Score = AUPRC) on miRBench standard benchmark datasets.

Calls the existing run_mirbench_standard_benchmark() which evaluates our model + 11 baselines
on 3 standard test sets: Klimentova, Manakov, CLASH.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from deepmirt.evaluation.comparison import run_mirbench_standard_benchmark


def main():
    ckpt_path = str(project_root / "checkpoints" / "epoch=27-val_auroc=0.9612.ckpt")
    config_path = str(project_root / "deepmirt" / "configs" / "default.yaml")

    logger.info("Starting miRBench standard benchmark evaluation...")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Config: {config_path}")

    results = run_mirbench_standard_benchmark(
        our_ckpt_path=ckpt_path,
        our_config_path=config_path,
        device="cuda:0",
        batch_size=256,
    )

    if not results:
        logger.error("No results returned. Check miRBench installation.")
        sys.exit(1)

    # Print and save results
    output_dir = project_root / "paper" / "mirbench_aps"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, df in results.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 80}")

        # Show key columns
        display_cols = ["Method", "Type", "AUROC", "AUPRC", "F1", "MCC"]
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].to_string(index=False, float_format="%.4f"))

        # Save CSV
        csv_path = output_dir / f"{ds_name}_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved to {csv_path}")

    # Print summary table (APS only, for quick comparison with miRBench paper Table 2)
    print(f"\n{'=' * 80}")
    print("SUMMARY: APS (AUPRC) across datasets — for comparison with miRBench paper Table 2")
    print(f"{'=' * 80}")

    all_methods = set()
    for df in results.values():
        all_methods.update(df["Method"].tolist())

    summary_rows = []
    for method in sorted(all_methods):
        row = {"Method": method}
        for ds_name, df in results.items():
            short_name = ds_name.replace("AGO2_", "").replace("_", " ")
            match = df[df["Method"] == method]
            if not match.empty:
                row[f"APS ({short_name})"] = match.iloc[0]["AUPRC"]
                row[f"AUROC ({short_name})"] = match.iloc[0]["AUROC"]
        summary_rows.append(row)

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    summary_path = output_dir / "summary_aps.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
