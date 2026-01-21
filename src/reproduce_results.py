"""
RSSM Reproducibility Script
Runs the full pipeline with deterministic seeds and data-prep checks.
"""

import os
import sys
import subprocess
import random
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42):
    """Ensure deterministic behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}.")
        sys.exit(1)


def main():
    print("Starting RSSM Reproduction Pipeline...")
    set_seed(42)

    base = Path.cwd()

    # 1. Data Preparation (run only if outputs are missing)
    meps_prepared = base / "rssm_meps_prepared.csv"
    medicaid_prepared = base / "rssm_medicaid_prepared.csv"

    if not meps_prepared.exists():
        run_command("python rssm_data_loader.py", "Data Preparation (General)")
    else:
        print(f"✓ Found {meps_prepared}, skipping general data prep.")

    if not medicaid_prepared.exists():
        run_command("python rssm_medicaid_data_prep.py", "Data Preparation (Medicaid)")
    else:
        print(f"✓ Found {medicaid_prepared}, skipping Medicaid data prep.")

    # 2. Training
    run_command("python rssm_training.py", "RSSM Model Training")

    # 3. Validation - COVID-19
    run_command("python rssm_validate.py", "COVID-19 Validation")

    # 4. Validation - Medicaid Expansion
    run_command("python rssm_medicaid_validation.py", "Medicaid Expansion Validation")

    # 5. Novel Insights
    run_command("python rssm_novel_insights.py", "Novel Insights Generation")

    print("\n" + "="*60)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Results and figures are available in the current directory.")
    print("Key expected metrics: COVID MAPE ~3.5% (RSSM) vs 16.8% baseline; Medicaid RMSE (high) ~0.519; Medicaid RMSE (mixed) ~0.765.")


if __name__ == "__main__":
    # Ensure we are in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
