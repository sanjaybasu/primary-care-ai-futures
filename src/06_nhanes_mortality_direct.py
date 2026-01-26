#!/usr/bin/env python3
"""
Direct analysis of NHANES mortality data without additional covariates.

We have 59,064 individuals with mortality follow-up. Let's analyze what we can
from the mortality file itself, which includes cause of death indicators.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDIVIDUAL_DIR = os.path.join(DATA_DIR, 'individual')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("=" * 70)
print("NHANES MORTALITY ANALYSIS - DIRECT FROM LINKED FILES")
print("=" * 70)

# Load mortality data
mort_df = pd.read_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_mortality_linked.csv'),
                       dtype={'SEQN': str})

# Convert columns
for col in ['ELIGSTAT', 'MORTSTAT', 'DODYEAR', 'UCOD_LEADING', 'DIABETES', 'HYPERTEN']:
    mort_df[col] = pd.to_numeric(mort_df[col], errors='coerce')

# Filter to eligible
mort_df = mort_df[mort_df['ELIGSTAT'] == 1].copy()
mort_df['died'] = (mort_df['MORTSTAT'] == 1).astype(int)

print(f"\nSample: {len(mort_df)} individuals eligible for mortality follow-up")
print(f"Deaths: {mort_df['died'].sum()}")
print(f"Mortality: {mort_df['died'].mean()*100:.2f}%")

# ============================================================================
# ANALYSIS BY SURVEY CYCLE (TEMPORAL TRENDS)
# ============================================================================
print("\n" + "=" * 50)
print("MORTALITY BY SURVEY CYCLE")
print("=" * 50)

cycle_stats = mort_df.groupby('survey_cycle').agg({
    'SEQN': 'count',
    'died': ['sum', 'mean']
}).round(4)
cycle_stats.columns = ['N', 'Deaths', 'Mortality_Rate']
cycle_stats = cycle_stats.sort_index()

print("\n  Survey Cycle       N    Deaths  Mortality")
print("  " + "-" * 45)
for idx, row in cycle_stats.iterrows():
    print(f"  {idx}       {int(row['N']):5d}    {int(row['Deaths']):4d}    {row['Mortality_Rate']*100:6.2f}%")

# ============================================================================
# CAUSE OF DEATH ANALYSIS
# ============================================================================
print("\n" + "=" * 50)
print("CAUSE OF DEATH ANALYSIS")
print("=" * 50)

# UCOD_LEADING codes (from NCHS documentation):
# 1 = Diseases of heart
# 2 = Malignant neoplasms
# 3 = Chronic lower respiratory diseases
# 4 = Accidents (unintentional injuries)
# 5 = Cerebrovascular diseases
# 6 = Alzheimer's disease
# 7 = Diabetes mellitus
# 8 = Influenza and pneumonia
# 9 = Nephritis, nephrotic syndrome, nephrosis
# 10 = All other causes

cod_labels = {
    1: 'Heart disease',
    2: 'Cancer',
    3: 'Chronic respiratory',
    4: 'Accidents',
    5: 'Cerebrovascular',
    6: 'Alzheimer\'s',
    7: 'Diabetes',
    8: 'Flu/pneumonia',
    9: 'Kidney disease',
    10: 'Other causes'
}

deaths = mort_df[mort_df['died'] == 1].copy()
cod_counts = deaths['UCOD_LEADING'].value_counts().sort_index()

print("\n  Cause of Death          N    Percent")
print("  " + "-" * 40)
for code, count in cod_counts.items():
    label = cod_labels.get(code, f'Code {code}')
    pct = count / len(deaths) * 100
    print(f"  {label:<22} {count:4d}   {pct:5.1f}%")

# Primary care amenable conditions
pc_amenable = deaths[deaths['UCOD_LEADING'].isin([1, 3, 5, 7, 8, 9])]
print(f"\n  Primary care amenable deaths: {len(pc_amenable)} ({len(pc_amenable)/len(deaths)*100:.1f}%)")

# ============================================================================
# DIABETES AND HYPERTENSION FLAGS
# ============================================================================
print("\n" + "=" * 50)
print("CHRONIC CONDITION FLAGS FROM DEATH CERTIFICATE")
print("=" * 50)

# DIABETES and HYPERTEN are flags indicating these conditions on death certificate
# 0 = No, 1 = Yes

diabetes_deaths = deaths[deaths['DIABETES'] == 1]
hypertension_deaths = deaths[deaths['HYPERTEN'] == 1]

print(f"\n  Diabetes on death certificate: {len(diabetes_deaths)} ({len(diabetes_deaths)/len(deaths)*100:.1f}%)")
print(f"  Hypertension on death certificate: {len(hypertension_deaths)} ({len(hypertension_deaths)/len(deaths)*100:.1f}%)")

# ============================================================================
# TEMPORAL ANALYSIS - YEAR OF DEATH
# ============================================================================
print("\n" + "=" * 50)
print("YEAR OF DEATH DISTRIBUTION")
print("=" * 50)

year_counts = deaths['DODYEAR'].value_counts().sort_index()
print("\n  Year    Deaths")
print("  " + "-" * 15)
for year, count in year_counts.items():
    if pd.notna(year) and year > 0:
        print(f"  {int(year)}    {count:4d}")

# ============================================================================
# TRY TO DOWNLOAD NHANES DEMO DATA USING DIFFERENT METHOD
# ============================================================================
print("\n" + "=" * 50)
print("DOWNLOADING NHANES DEMOGRAPHICS (ALTERNATE METHOD)")
print("=" * 50)

import urllib.request
import tempfile

# Try direct download to temp file then read
demo_files_direct = {
    '2017-2018': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT',
    '2015-2016': 'https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT',
}

all_demo = []

for cycle, url in demo_files_direct.items():
    print(f"\n  Trying {cycle}...")
    try:
        # Download to temp file
        temp_path = os.path.join(RAW_DIR, f'demo_{cycle.replace("-", "_")}.xpt')
        urllib.request.urlretrieve(url, temp_path)
        filesize = os.path.getsize(temp_path)
        print(f"    Downloaded: {filesize/1024:.1f} KB")

        # Try reading with pandas
        try:
            demo = pd.read_sas(temp_path, format='xport')
            demo['survey_cycle'] = cycle.replace('-', '_')
            all_demo.append(demo)
            print(f"    Loaded: {len(demo)} records")
            print(f"    Columns: {list(demo.columns[:10])}...")
        except Exception as e:
            print(f"    Read error: {e}")

    except Exception as e:
        print(f"    Download error: {e}")

if all_demo:
    demo_df = pd.concat(all_demo, ignore_index=True)
    print(f"\n  Total demographic records: {len(demo_df)}")

    # Save for later use
    demo_df.to_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_demographics.csv'), index=False)

    # Try to merge with mortality
    demo_df['SEQN'] = demo_df['SEQN'].astype(int).astype(str)
    mort_df['SEQN'] = mort_df['SEQN'].str.strip()

    # Match on SEQN (need to handle the cycle matching)
    merged = mort_df.merge(demo_df, on='SEQN', how='inner')
    print(f"  Merged records: {len(merged)}")

    if len(merged) > 0 and 'RIDAGEYR' in merged.columns:
        print(f"\n  Age range: {merged['RIDAGEYR'].min():.0f} - {merged['RIDAGEYR'].max():.0f}")
        print(f"  Mean age: {merged['RIDAGEYR'].mean():.1f}")

        # Mortality by age group
        merged['age_group'] = pd.cut(merged['RIDAGEYR'],
                                      bins=[0, 45, 65, 120],
                                      labels=['18-44', '45-64', '65+'])

        print("\n  Mortality by Age Group:")
        for ag in ['18-44', '45-64', '65+']:
            subset = merged[merged['age_group'] == ag]
            if len(subset) > 0:
                print(f"    {ag}: {subset['died'].mean()*100:.2f}% (n={len(subset)}, deaths={subset['died'].sum()})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("REAL DATA SUMMARY")
print("=" * 70)

print(f"""
NHANES PUBLIC-USE LINKED MORTALITY FILES
=========================================

Data Source: NCHS Public-Use Linked Mortality Files
URL: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

Sample Size: {len(mort_df):,} individuals
Survey Cycles: 1999-2000 through 2017-2018
Follow-up: Through December 31, 2019

MORTALITY OUTCOMES:
  Total deaths: {mort_df['died'].sum():,}
  Crude mortality: {mort_df['died'].mean()*100:.2f}%
  Person-years (approximate): {len(mort_df) * 10:,} (assuming ~10 year average follow-up)

CAUSE OF DEATH (Top 5):
  Heart disease: {cod_counts.get(1, 0):,} ({cod_counts.get(1, 0)/len(deaths)*100:.1f}%)
  Cancer: {cod_counts.get(2, 0):,} ({cod_counts.get(2, 0)/len(deaths)*100:.1f}%)
  Chronic respiratory: {cod_counts.get(3, 0):,} ({cod_counts.get(3, 0)/len(deaths)*100:.1f}%)
  Accidents: {cod_counts.get(4, 0):,} ({cod_counts.get(4, 0)/len(deaths)*100:.1f}%)
  Cerebrovascular: {cod_counts.get(5, 0):,} ({cod_counts.get(5, 0)/len(deaths)*100:.1f}%)

PRIMARY CARE AMENABLE:
  Deaths from PC-amenable conditions: {len(pc_amenable):,} ({len(pc_amenable)/len(deaths)*100:.1f}%)
  (Heart disease, respiratory, stroke, diabetes, flu/pneumonia, kidney)

CHRONIC CONDITIONS (from death certificate):
  Diabetes mentioned: {len(diabetes_deaths):,} ({len(diabetes_deaths)/len(deaths)*100:.1f}%)
  Hypertension mentioned: {len(hypertension_deaths):,} ({len(hypertension_deaths)/len(deaths)*100:.1f}%)
""")

# Save summary
summary_data = {
    'metric': ['sample_size', 'deaths', 'mortality_rate', 'person_years_approx',
               'heart_disease_deaths', 'cancer_deaths', 'pc_amenable_deaths',
               'diabetes_on_dc', 'hypertension_on_dc'],
    'value': [len(mort_df), mort_df['died'].sum(), mort_df['died'].mean(),
              len(mort_df) * 10,
              cod_counts.get(1, 0), cod_counts.get(2, 0), len(pc_amenable),
              len(diabetes_deaths), len(hypertension_deaths)]
}

pd.DataFrame(summary_data).to_csv(os.path.join(RESULTS_DIR, 'nhanes_mortality_summary.csv'), index=False)
print(f"\nResults saved to: {RESULTS_DIR}")
