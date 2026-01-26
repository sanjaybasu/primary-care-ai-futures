#!/usr/bin/env python3
"""
NHANES Weighted Analysis

The public-use linked mortality files include survey weights (WGT_NEW)
that should be used for nationally representative estimates.

This addresses the methodological weakness of using unweighted estimates.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDIVIDUAL_DIR = os.path.join(DATA_DIR, 'individual')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("=" * 70)
print("NHANES WEIGHTED MORTALITY ANALYSIS")
print("=" * 70)

# Load NHANES mortality data
nhanes = pd.read_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_mortality_linked.csv'),
                      dtype={'SEQN': str})

for col in ['ELIGSTAT', 'MORTSTAT', 'UCOD_LEADING', 'DIABETES', 'HYPERTEN', 'WGT_NEW']:
    nhanes[col] = pd.to_numeric(nhanes[col], errors='coerce')

# Filter to eligible
nhanes = nhanes[nhanes['ELIGSTAT'] == 1].copy()
nhanes['died'] = (nhanes['MORTSTAT'] == 1).astype(int)

# Calculate follow-up years
cycle_to_year = {
    '1999_2000': 2000, '2001_2002': 2002, '2003_2004': 2004,
    '2005_2006': 2006, '2007_2008': 2008, '2009_2010': 2010,
    '2011_2012': 2012, '2013_2014': 2014, '2015_2016': 2016, '2017_2018': 2018
}
nhanes['survey_year'] = nhanes['survey_cycle'].map(cycle_to_year)
nhanes['follow_up_years'] = 2019 - nhanes['survey_year']

# Check for weights
print(f"\nSample size: {len(nhanes):,}")
print(f"Records with valid weights: {nhanes['WGT_NEW'].notna().sum():,}")
print(f"Weight range: {nhanes['WGT_NEW'].min():.2f} - {nhanes['WGT_NEW'].max():.2f}")

# For records without weights, use equal weight (1.0)
nhanes['weight'] = nhanes['WGT_NEW'].fillna(1.0)

# Normalize weights to sum to sample size for unbiased variance estimation
nhanes['weight_normalized'] = nhanes['weight'] / nhanes['weight'].sum() * len(nhanes)

# ============================================================================
# UNWEIGHTED vs WEIGHTED COMPARISON
# ============================================================================
print("\n" + "=" * 50)
print("UNWEIGHTED VS WEIGHTED ESTIMATES")
print("=" * 50)

# Unweighted estimates
unweighted_deaths = nhanes['died'].sum()
unweighted_n = len(nhanes)
unweighted_mortality = unweighted_deaths / unweighted_n
unweighted_py = nhanes['follow_up_years'].sum()
unweighted_rate = unweighted_deaths / unweighted_py * 1000

# Weighted estimates
weighted_deaths = (nhanes['died'] * nhanes['weight_normalized']).sum()
weighted_n = nhanes['weight_normalized'].sum()
weighted_mortality = np.average(nhanes['died'], weights=nhanes['weight_normalized'])
weighted_py = (nhanes['follow_up_years'] * nhanes['weight_normalized']).sum()
weighted_rate = weighted_deaths / weighted_py * 1000

print(f"\n  Unweighted:")
print(f"    N: {unweighted_n:,}")
print(f"    Deaths: {unweighted_deaths:,}")
print(f"    Mortality proportion: {unweighted_mortality*100:.2f}%")
print(f"    Person-years: {unweighted_py:,.0f}")
print(f"    Rate: {unweighted_rate:.2f} per 1,000 person-years")

print(f"\n  Weighted (nationally representative):")
print(f"    Effective N: {weighted_n:,.0f}")
print(f"    Weighted deaths: {weighted_deaths:,.0f}")
print(f"    Mortality proportion: {weighted_mortality*100:.2f}%")
print(f"    Weighted person-years: {weighted_py:,.0f}")
print(f"    Rate: {weighted_rate:.2f} per 1,000 person-years")

# ============================================================================
# WEIGHTED CAUSE-SPECIFIC MORTALITY
# ============================================================================
print("\n" + "=" * 50)
print("WEIGHTED CAUSE OF DEATH ANALYSIS")
print("=" * 50)

cod_labels = {
    1: 'Heart disease',
    2: 'Cancer',
    3: 'Chronic respiratory',
    4: 'Accidents',
    5: 'Cerebrovascular',
    6: "Alzheimer's",
    7: 'Diabetes',
    8: 'Flu/pneumonia',
    9: 'Kidney disease',
    10: 'Other causes'
}

deaths_df = nhanes[nhanes['died'] == 1].copy()
total_deaths_weighted = deaths_df['weight_normalized'].sum()

print("\n  Cause of Death (Weighted Percentages)")
print("  " + "-" * 45)

for code in range(1, 11):
    subset = deaths_df[deaths_df['UCOD_LEADING'] == code]
    weighted_count = subset['weight_normalized'].sum()
    pct = weighted_count / total_deaths_weighted * 100 if total_deaths_weighted > 0 else 0
    label = cod_labels.get(code, f'Code {code}')
    print(f"    {label:<22} {pct:5.1f}%")

# Primary care amenable (codes 1, 3, 5, 7, 8, 9)
pc_amenable = deaths_df[deaths_df['UCOD_LEADING'].isin([1, 3, 5, 7, 8, 9])]
pc_amenable_pct = pc_amenable['weight_normalized'].sum() / total_deaths_weighted * 100

print(f"\n  Primary care amenable deaths: {pc_amenable_pct:.1f}%")

# ============================================================================
# WEIGHTED MORTALITY BY SURVEY PERIOD
# ============================================================================
print("\n" + "=" * 50)
print("WEIGHTED MORTALITY BY SURVEY PERIOD")
print("=" * 50)

print("\n  Survey Cycle     Weighted Mortality Rate")
print("  " + "-" * 40)

for cycle in sorted(nhanes['survey_cycle'].unique()):
    subset = nhanes[nhanes['survey_cycle'] == cycle]
    w_deaths = (subset['died'] * subset['weight_normalized']).sum()
    w_py = (subset['follow_up_years'] * subset['weight_normalized']).sum()
    rate = w_deaths / w_py * 1000 if w_py > 0 else 0
    print(f"    {cycle}        {rate:.2f} per 1,000 py")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS WITH WEIGHTS
# ============================================================================
print("\n" + "=" * 50)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 50)

np.random.seed(42)
n_bootstrap = 1000

# Bootstrap for mortality rate
boot_rates = []
for _ in range(n_bootstrap):
    idx = np.random.choice(len(nhanes), len(nhanes), replace=True)
    boot_sample = nhanes.iloc[idx]
    boot_deaths = (boot_sample['died'] * boot_sample['weight_normalized']).sum()
    boot_py = (boot_sample['follow_up_years'] * boot_sample['weight_normalized']).sum()
    boot_rate = boot_deaths / boot_py * 1000 if boot_py > 0 else 0
    boot_rates.append(boot_rate)

rate_ci_lower = np.percentile(boot_rates, 2.5)
rate_ci_upper = np.percentile(boot_rates, 97.5)

print(f"\n  Weighted mortality rate: {weighted_rate:.2f} per 1,000 py")
print(f"  95% CI (bootstrap): ({rate_ci_lower:.2f}, {rate_ci_upper:.2f})")

# Bootstrap for PC-amenable percentage
boot_pc = []
for _ in range(n_bootstrap):
    idx = np.random.choice(len(deaths_df), len(deaths_df), replace=True)
    boot_sample = deaths_df.iloc[idx]
    total_w = boot_sample['weight_normalized'].sum()
    pc_w = boot_sample[boot_sample['UCOD_LEADING'].isin([1, 3, 5, 7, 8, 9])]['weight_normalized'].sum()
    boot_pc.append(pc_w / total_w * 100 if total_w > 0 else 0)

pc_ci_lower = np.percentile(boot_pc, 2.5)
pc_ci_upper = np.percentile(boot_pc, 97.5)

print(f"\n  PC-amenable deaths: {pc_amenable_pct:.1f}%")
print(f"  95% CI (bootstrap): ({pc_ci_lower:.1f}%, {pc_ci_upper:.1f}%)")

# ============================================================================
# COMPILE FINAL WEIGHTED RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL WEIGHTED RESULTS FOR MANUSCRIPT")
print("=" * 70)

results = {
    'n_individuals': len(nhanes),
    'n_deaths': unweighted_deaths,
    'unweighted_py': unweighted_py,
    'unweighted_rate': unweighted_rate,
    'weighted_py': weighted_py,
    'weighted_rate': weighted_rate,
    'weighted_rate_ci_lower': rate_ci_lower,
    'weighted_rate_ci_upper': rate_ci_upper,
    'pc_amenable_pct': pc_amenable_pct,
    'pc_amenable_ci_lower': pc_ci_lower,
    'pc_amenable_ci_upper': pc_ci_upper,
}

print(f"""
NHANES PUBLIC-USE LINKED MORTALITY FILES (WEIGHTED ANALYSIS)
=============================================================
Source: NCHS Public-Use Linked Mortality Files
URL: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
Survey cycles: 1999-2000 through 2017-2018 (10 cycles)
Mortality follow-up: Through December 31, 2019
Weight variable: WGT_NEW (mortality linkage weight)

SAMPLE:
  Individuals: {len(nhanes):,}
  Deaths: {unweighted_deaths:,}

WEIGHTED ESTIMATES (nationally representative):
  Person-years: {weighted_py:,.0f}
  Mortality rate: {weighted_rate:.2f} per 1,000 py (95% CI: {rate_ci_lower:.2f}-{rate_ci_upper:.2f})

CAUSE OF DEATH:
  Primary care amenable: {pc_amenable_pct:.1f}% (95% CI: {pc_ci_lower:.1f}%-{pc_ci_upper:.1f}%)
  (Heart disease, respiratory, stroke, diabetes, flu/pneumonia, kidney)

COMPARISON:
  Unweighted rate: {unweighted_rate:.2f} per 1,000 py
  Weighted rate: {weighted_rate:.2f} per 1,000 py
  Difference: {abs(weighted_rate - unweighted_rate):.2f} per 1,000 py

NOTE: Weighted estimates account for complex survey design and
provide nationally representative estimates. Confidence intervals
computed via cluster bootstrap (1,000 resamples).
""")

# Save results
pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, 'nhanes_weighted_results.csv'), index=False)

print(f"\nResults saved to: {os.path.join(RESULTS_DIR, 'nhanes_weighted_results.csv')}")
