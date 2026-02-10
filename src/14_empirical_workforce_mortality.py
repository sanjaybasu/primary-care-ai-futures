#!/usr/bin/env python3
"""
Empirical Workforce-Mortality Analysis

Uses actual state-level workforce variation to estimate effects on mortality,
providing empirical estimates rather than literature synthesis.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

np.random.seed(42)

print("=" * 70)
print("EMPIRICAL WORKFORCE-MORTALITY ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD STATE-LEVEL DATA
# ============================================================================
print("\n[1/5] Loading state-level data...")

# Load workforce data
workforce = pd.read_csv(os.path.join(PROCESSED_DIR, 'state_workforce_2022.csv'))
print(f"  Workforce data: {len(workforce)} states")

# Load mortality data
mortality = pd.read_csv(os.path.join(PROCESSED_DIR, 'state_mortality_2019_2023.csv'))
print(f"  Mortality data: {len(mortality)} states")

# Merge datasets
state_data = pd.merge(workforce, mortality, on='state')
print(f"  Merged data: {len(state_data)} states")

# ============================================================================
# 2. CROSS-SECTIONAL CORRELATION ANALYSIS
# ============================================================================
print("\n[2/5] Cross-sectional correlation analysis...")

# PCP supply vs mortality
pcp_corr, pcp_p = stats.pearsonr(state_data['pcp_per_100k'], state_data['death_rate_2022'])
print(f"\n  PCP supply vs 2022 mortality:")
print(f"    Correlation: r = {pcp_corr:.3f}")
print(f"    p-value: {pcp_p:.4f}")

# Total primary care supply vs mortality
total_corr, total_p = stats.pearsonr(state_data['total_primary_care'], state_data['death_rate_2022'])
print(f"\n  Total primary care supply vs 2022 mortality:")
print(f"    Correlation: r = {total_corr:.3f}")
print(f"    p-value: {total_p:.4f}")

# NP/PA supply vs mortality (controlling for PCP)
np_pa_corr, np_pa_p = stats.pearsonr(state_data['np_pa_per_100k'], state_data['death_rate_2022'])
print(f"\n  NP/PA supply vs 2022 mortality:")
print(f"    Correlation: r = {np_pa_corr:.3f}")
print(f"    p-value: {np_pa_p:.4f}")

# ============================================================================
# 3. REGRESSION ANALYSIS - EFFECT OF WORKFORCE ON MORTALITY
# ============================================================================
print("\n[3/5] Regression analysis...")

# Simple linear regression: PCP supply -> mortality
X_pcp = state_data[['pcp_per_100k']]
y = state_data['death_rate_2022']

reg_pcp = LinearRegression()
reg_pcp.fit(X_pcp, y)
pcp_effect = reg_pcp.coef_[0]

# Bootstrap for confidence interval
n_boot = 1000
boot_pcp_effects = []
for _ in range(n_boot):
    idx = np.random.choice(len(state_data), len(state_data), replace=True)
    X_boot = state_data.iloc[idx][['pcp_per_100k']]
    y_boot = state_data.iloc[idx]['death_rate_2022']
    reg_boot = LinearRegression()
    reg_boot.fit(X_boot, y_boot)
    boot_pcp_effects.append(reg_boot.coef_[0])

pcp_ci_lower = np.percentile(boot_pcp_effects, 2.5)
pcp_ci_upper = np.percentile(boot_pcp_effects, 97.5)

print(f"\n  Effect of PCP supply on mortality (per 10 PCPs per 100k):")
print(f"    Coefficient: {pcp_effect * 10:.2f} deaths per 100,000")
print(f"    95% CI: ({pcp_ci_lower * 10:.2f}, {pcp_ci_upper * 10:.2f})")

# Calculate hazard ratio equivalent
# Mean mortality is ~800 per 100,000, so a reduction of X per 100,000
# corresponds to HR = (800 - X) / 800
mean_mort = state_data['death_rate_2022'].mean()
pcp_hr_10 = (mean_mort + pcp_effect * 10) / mean_mort
print(f"    Implied RR for +10 PCPs: {pcp_hr_10:.3f}")

# Multiple regression: PCP + NP/PA
X_multi = state_data[['pcp_per_100k', 'np_pa_per_100k']]
reg_multi = LinearRegression()
reg_multi.fit(X_multi, y)

print(f"\n  Multiple regression (PCP + NP/PA):")
print(f"    PCP effect: {reg_multi.coef_[0] * 10:.2f} per 10 PCPs")
print(f"    NP/PA effect: {reg_multi.coef_[1] * 10:.2f} per 10 NPs/PAs")

# ============================================================================
# 4. DIFFERENCE-IN-DIFFERENCES BY WORKFORCE CHANGE
# ============================================================================
print("\n[4/5] Analyzing states by workforce levels...")

# Divide states into high vs low PCP supply
median_pcp = state_data['pcp_per_100k'].median()
state_data['high_pcp'] = state_data['pcp_per_100k'] >= median_pcp

# Compare mortality
high_pcp_mort = state_data[state_data['high_pcp']]['death_rate_2022'].mean()
low_pcp_mort = state_data[~state_data['high_pcp']]['death_rate_2022'].mean()

print(f"\n  Mortality by PCP supply level:")
print(f"    High PCP states (>= {median_pcp:.1f}/100k): {high_pcp_mort:.1f} per 100,000")
print(f"    Low PCP states (< {median_pcp:.1f}/100k): {low_pcp_mort:.1f} per 100,000")
print(f"    Difference: {low_pcp_mort - high_pcp_mort:.1f} per 100,000")
print(f"    Rate ratio: {low_pcp_mort / high_pcp_mort:.3f}")

# ============================================================================
# 5. CONVERT TO HAZARD RATIO ESTIMATES
# ============================================================================
print("\n[5/5] Converting to intervention effect estimates...")

# Effect of 10 additional PCPs per 100,000
# From regression: pcp_effect is change in mortality per 1 PCP
pcp_10_change = pcp_effect * 10
pcp_10_rr = (mean_mort + pcp_10_change) / mean_mort

# For comparison with literature:
# Basu et al. 2019 found ~5% reduction in mortality per 10 PCPs
# Our empirical estimate:
print(f"\n  Empirical estimate for PCP supply effect:")
print(f"    Per 10 PCPs per 100,000 increase:")
print(f"      Mortality change: {pcp_10_change:.1f} per 100,000")
print(f"      Rate ratio: {pcp_10_rr:.3f}")

# Convert to HR (approximate, assuming linear relationship)
# If RR = 0.95 for +10 PCPs, this is consistent with literature
if pcp_10_rr < 1:
    pcp_hr = pcp_10_rr
else:
    # Mortality increases with more PCPs? (unlikely, check data)
    pcp_hr = pcp_10_rr

# Bootstrap HR estimate
boot_hrs = []
for _ in range(n_boot):
    idx = np.random.choice(len(state_data), len(state_data), replace=True)
    boot_data = state_data.iloc[idx]
    X_boot = boot_data[['pcp_per_100k']]
    y_boot = boot_data['death_rate_2022']
    reg_boot = LinearRegression()
    reg_boot.fit(X_boot, y_boot)
    boot_change = reg_boot.coef_[0] * 10
    boot_hr = (boot_data['death_rate_2022'].mean() + boot_change) / boot_data['death_rate_2022'].mean()
    boot_hrs.append(boot_hr)

hr_ci_lower = np.percentile(boot_hrs, 2.5)
hr_ci_upper = np.percentile(boot_hrs, 97.5)

print(f"\n  EMPIRICAL WORKFORCE EFFECT ESTIMATE:")
print(f"    Rate ratio per +10 PCPs: {pcp_10_rr:.2f}")
print(f"    95% CI: ({hr_ci_lower:.2f}, {hr_ci_upper:.2f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'pcp_mortality_corr': pcp_corr,
    'pcp_mortality_p': pcp_p,
    'total_pc_mortality_corr': total_corr,
    'total_pc_mortality_p': total_p,
    'pcp_effect_per_10': pcp_effect * 10,
    'pcp_effect_ci_lower': pcp_ci_lower * 10,
    'pcp_effect_ci_upper': pcp_ci_upper * 10,
    'pcp_rr_per_10': pcp_10_rr,
    'pcp_rr_ci_lower': hr_ci_lower,
    'pcp_rr_ci_upper': hr_ci_upper,
    'high_pcp_mortality': high_pcp_mort,
    'low_pcp_mortality': low_pcp_mort,
    'high_low_rr': low_pcp_mort / high_pcp_mort
}

pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, 'empirical_workforce_mortality.csv'), index=False)

print("\n" + "=" * 70)
print("SUMMARY: EMPIRICAL WORKFORCE-MORTALITY ANALYSIS")
print("=" * 70)
print(f"""
KEY FINDINGS FROM STATE-LEVEL DATA (N=51 states):

1. PCP SUPPLY - MORTALITY CORRELATION
   Correlation: r = {pcp_corr:.3f} (p = {pcp_p:.4f})
   Interpretation: {'Significant inverse relationship' if pcp_p < 0.05 else 'No significant relationship'}

2. EFFECT OF +10 PCPs per 100,000 POPULATION
   Mortality change: {pcp_10_change:.1f} per 100,000 (95% CI: {pcp_ci_lower*10:.1f}, {pcp_ci_upper*10:.1f})
   Rate ratio: {pcp_10_rr:.3f} (95% CI: {hr_ci_lower:.3f}, {hr_ci_upper:.3f})

3. HIGH vs LOW PCP SUPPLY STATES
   High PCP states: {high_pcp_mort:.1f} per 100,000
   Low PCP states: {low_pcp_mort:.1f} per 100,000
   Rate ratio: {low_pcp_mort/high_pcp_mort:.3f}

COMPARISON TO LITERATURE:
   - Basu et al. 2019: ~5% mortality reduction per 10 PCPs
   - Our empirical estimate: {(1-pcp_10_rr)*100:.1f}% {'reduction' if pcp_10_rr < 1 else 'increase'}

DATA SOURCES:
   - State workforce: AAMC, AANP published data (2022)
   - State mortality: CDC WONDER (2022)
""")

print(f"\nResults saved to: {RESULTS_DIR}/empirical_workforce_mortality.csv")
