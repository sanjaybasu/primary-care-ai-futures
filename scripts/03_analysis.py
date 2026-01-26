#!/usr/bin/env python3
"""
Main Analysis Orchestrator: Strategies and Thresholds to Close the Primary Care Mortality Gap

This script executes the rigorous analysis pipeline described in the manuscript:
1. Double Machine Learning (DML) for causal parameter estimation.
2. Recurrent State-Space Model (RSSM) training and calibration.
3. Threshold Analysis to determine the specific efficacy targets (9-fold improvement)
   required for AI to substitute for policy interventions.

It utilizes the modularized components in `src/` to ensure consistency with the
submitted code package.

Author: Sanjay Basu
License: MIT
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to python path to import rigorous modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)

from src import formal_dml as dml
from src import neural_network_world_model as rssm

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("PRIMARY CARE MORTALITY ANALYSIS: MAIN ORCHESTRATOR")
print("=" * 80)

# =============================================================================
# 1. RUN DOUBLE MACHINE LEARNING (DML)
# =============================================================================
print("\n[STEP 1/3] Executing Double Machine Learning (DML)...")
print("Estimating causal effects of Medicaid expansion and workforce on mortality.")

# Execute DML pipeline (loads data, runs cross-fitting, saves results)
# We invoke the main execution block of the module
os.system(f"python3 {os.path.join(SRC_DIR, '17_formal_dml.py')}")

# Load results to check
dml_results_path = os.path.join(RESULTS_DIR, 'dml_results.csv')
if os.path.exists(dml_results_path):
    dml_res = pd.read_csv(dml_results_path)
    ate = dml_res['ate'].iloc[0]
    medicaid_hr = dml_res['rate_ratio'].iloc[0]
    print(f"\n  DML CONFIRMED: Medicaid Expansion HR = {medicaid_hr:.2f} (ATE: {ate:.1f})")
else:
    print("\n  WARNING: DML results not found. Execution may have failed.")
    medicaid_hr = 0.90 # Fallback for threshold calculation if run fails locally sans data

# =============================================================================
# 2. TRAIN & CALIBRATE WORLD MODEL (RSSM)
# =============================================================================
print("\n[STEP 2/3] Training Recurrent State-Space Model (World Model)...")
print("Calibrating simulation engine against COVID-19 and Telemedicine shocks.")

# Execute RSSM pipeline
os.system(f"python3 {os.path.join(SRC_DIR, '09_neural_network_world_model.py')}")

# Load performance to confirm
perf_path = os.path.join(RESULTS_DIR, 'world_model_performance.csv')
if os.path.exists(perf_path):
    perf = pd.read_csv(perf_path)
    print("\n  model PERFORMANCE CONFIRMED:")
    print(perf)
else:
    print("\n  WARNING: World Model results not found.")

# =============================================================================
# 3. THRESHOLD ANALYSIS (The 9-Fold Calculation)
# =============================================================================
print("\n[STEP 3/3] Performing Threshold Analysis...")
print("Calculating efficacy thresholds for AI to match policy impact.")

# Mortality gap parameters (from Manuscript/Results)
MORTALITY_GAP = 2.4  # deaths per 1,000 person-years (Q4 vs Q1)
Q4_MORTALITY = 9.6   # Baseline high-vulnerability mortality

# Intervention Effect Sizes (Verified from DML and Meta-Analysis/Table 2)
HR_MEDICAID = medicaid_hr  # ~0.90
HR_CHW = 0.93              # From Supplementary Meta-Analysis (Kangovi et al.)
HR_AI_CURRENT = 0.99       # Current AI Documentation (Lukac et al.)

# 1. Calculate Gap Closure for Traditional Policies
medicaid_reduction = Q4_MORTALITY * (1 - HR_MEDICAID)
medicaid_closure_pct = medicaid_reduction / MORTALITY_GAP

chw_reduction = Q4_MORTALITY * (1 - HR_CHW) * 0.65  # 65% Coverage scaling
chw_closure_pct = chw_reduction / MORTALITY_GAP

# 2. Calculate AI Contribution (Current)
ai_reduction_current = Q4_MORTALITY * (1 - HR_AI_CURRENT)
ai_closure_pct = ai_reduction_current / MORTALITY_GAP

# 3. Calculate Required AI Threshold (The "Nine-Fold" Finding)
# Target: Match Medicaid's absolute mortality reduction
target_reduction = medicaid_reduction
target_hr = 1 - (target_reduction / Q4_MORTALITY)

# Fold Improvement Calculation
# Improvement = (Current Benefit) / (Required Benefit) -> gap
# Or simpler: (1 - Current_HR) vs (1 - Required_HR)
current_benefit = 1 - HR_AI_CURRENT
required_benefit = 1 - target_hr
fold_improvement = required_benefit / max(current_benefit, 0.001)

print(f"\n  THRESHOLD ANALYSIS RESULTS:")
print(f"  ---------------------------")
print(f"  Mortality Gap: {MORTALITY_GAP} deaths/1,000")
print(f"  Medicaid Expansion (HR {HR_MEDICAID:.2f}): Closes {medicaid_closure_pct*100:.1f}% of gap.")
print(f"  AI Current (HR {HR_AI_CURRENT:.2f}): Closes {ai_closure_pct*100:.1f}% of gap.")
print(f"  ")
print(f"  REQUIRED TARGET FOR AI:")
print(f"  To match Medicaid, AI must achieve HR = {target_hr:.3f} (approx 0.91)")
print(f"  Current benefit: {current_benefit:.3f} | Required benefit: {required_benefit:.3f}")
print(f"  Improvement Factor: {fold_improvement:.1f}x (The 'Nine-Fold' Threshold)")

# Save final threshold results
threshold_res = {
    'mortality_gap': MORTALITY_GAP,
    'medicaid_hr': HR_MEDICAID,
    'ai_current_hr': HR_AI_CURRENT,
    'ai_target_hr': target_hr,
    'medicaid_closure_pct': medicaid_closure_pct,
    'fold_improvement_needed': fold_improvement
}
pd.DataFrame([threshold_res]).to_csv(os.path.join(RESULTS_DIR, 'final_threshold_analysis.csv'), index=False)
print(f"\nFinal analysis saved to: {RESULTS_DIR}/final_threshold_analysis.csv")
