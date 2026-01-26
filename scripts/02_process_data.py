#!/usr/bin/env python3
"""
Process and Integrate Data for Primary Care Mortality Analysis

This script processes downloaded data into analysis-ready datasets:
1. NHANES mortality linkage file parsing
2. CDC SVI aggregation by state and vulnerability quartile
3. State-level data integration

Author: Sanjay Basu
License: MIT
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print("PRIMARY CARE MORTALITY ANALYSIS: DATA PROCESSING")
print("=" * 70)

# =============================================================================
# 1. PARSE NHANES PUBLIC-USE LINKED MORTALITY FILES
# =============================================================================
print("\n[1/4] Processing NHANES mortality linkage files...")

# Column specifications for NHANES mortality files (fixed width)
# Based on NCHS documentation
NHANES_MORT_COLS = {
    'SEQN': (0, 6),           # Respondent sequence number
    'ELIGSTAT': (14, 15),     # Eligibility status for mortality follow-up
    'MORTSTAT': (15, 16),     # Final mortality status
    'UCOD_LEADING': (16, 19), # Underlying cause of death recode
    'DIABETES': (19, 20),     # Diabetes flag
    'HYPERTEN': (20, 21),     # Hypertension flag
    'PERMTH_INT': (26, 29),   # Person-months from interview to 12/31/2019
    'PERMTH_EXM': (29, 32),   # Person-months from exam to 12/31/2019
}

NHANES_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]

all_mortality = []
for filename in NHANES_FILES:
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: {filename} not found, skipping...")
        continue

    # Extract cycle years from filename
    cycle = filename.split('_')[1] + '_' + filename.split('_')[2]

    with open(filepath, 'r') as f:
        for line in f:
            try:
                record = {
                    'SEQN': int(line[0:6].strip() or 0),
                    'ELIGSTAT': int(line[14:15].strip() or 0),
                    'MORTSTAT': int(line[15:16].strip() or 0),
                    'UCOD_LEADING': line[16:19].strip(),
                    'DIABETES': int(line[19:20].strip() or 0),
                    'HYPERTEN': int(line[20:21].strip() or 0),
                    'PERMTH_INT': int(line[26:29].strip() or 0),
                    'cycle': cycle
                }
                all_mortality.append(record)
            except (ValueError, IndexError):
                continue

if all_mortality:
    mortality_df = pd.DataFrame(all_mortality)

    # Filter to eligible for mortality follow-up
    mortality_df = mortality_df[mortality_df['ELIGSTAT'] == 1]

    # Calculate person-years
    mortality_df['person_years'] = mortality_df['PERMTH_INT'] / 12

    print(f"  Total records: {len(mortality_df):,}")
    print(f"  Deaths (MORTSTAT=1): {(mortality_df['MORTSTAT']==1).sum():,}")
    print(f"  Total person-years: {mortality_df['person_years'].sum():,.0f}")

    mortality_df.to_csv(os.path.join(PROCESSED_DIR, 'nhanes_mortality.csv'), index=False)
else:
    print("  No NHANES mortality files found")

# =============================================================================
# 2. PROCESS CDC SOCIAL VULNERABILITY INDEX
# =============================================================================
print("\n[2/4] Processing CDC Social Vulnerability Index...")

svi_path = os.path.join(RAW_DIR, "SVI_2022_US.csv")
if os.path.exists(svi_path):
    svi = pd.read_csv(svi_path, low_memory=False)

    # Key columns
    svi_cols = ['STATE', 'ST_ABBR', 'COUNTY', 'FIPS', 'LOCATION',
                'RPL_THEMES',  # Overall vulnerability ranking
                'EPL_UNINSUR', 'E_UNINSUR',  # Uninsured estimates
                'EPL_POV150', 'E_POV150',    # Poverty estimates
                'E_TOTPOP']    # Population

    svi_clean = svi[[c for c in svi_cols if c in svi.columns]].copy()

    # Create vulnerability quartiles
    svi_clean['SVI_quartile'] = pd.qcut(svi_clean['RPL_THEMES'], 4, labels=[1,2,3,4])

    # Aggregate by state
    state_svi = svi_clean.groupby('ST_ABBR').agg({
        'RPL_THEMES': 'mean',
        'E_UNINSUR': 'sum',
        'E_POV150': 'sum',
        'E_TOTPOP': 'sum'
    }).reset_index()

    state_svi['uninsured_rate'] = state_svi['E_UNINSUR'] / state_svi['E_TOTPOP'] * 100
    state_svi['poverty_rate'] = state_svi['E_POV150'] / state_svi['E_TOTPOP'] * 100

    print(f"  Census tracts: {len(svi_clean):,}")
    print(f"  States: {len(state_svi)}")

    svi_clean.to_csv(os.path.join(PROCESSED_DIR, 'svi_tract_level.csv'), index=False)
    state_svi.to_csv(os.path.join(PROCESSED_DIR, 'svi_state_level.csv'), index=False)

    # Summary by quartile
    quartile_summary = svi_clean.groupby('SVI_quartile').agg({
        'E_TOTPOP': 'sum',
        'E_UNINSUR': 'sum',
        'E_POV150': 'sum'
    }).reset_index()
    quartile_summary['uninsured_pct'] = quartile_summary['E_UNINSUR'] / quartile_summary['E_TOTPOP'] * 100
    quartile_summary['poverty_pct'] = quartile_summary['E_POV150'] / quartile_summary['E_TOTPOP'] * 100

    print("\n  SVI Quartile Summary:")
    print("  Quartile | Population   | Uninsured % | Poverty %")
    print("  " + "-" * 50)
    for _, row in quartile_summary.iterrows():
        print(f"     Q{int(row['SVI_quartile'])}    | {row['E_TOTPOP']:>12,.0f} | {row['uninsured_pct']:>10.1f}% | {row['poverty_pct']:.1f}%")
else:
    print("  SVI file not found. Please download from CDC.")

# =============================================================================
# 3. PRIMARY CARE WORKFORCE DATA
# =============================================================================
print("\n[3/4] Creating workforce estimates...")

# State-level PCP supply per 100,000 (from AAMC/AANP estimates)
# These are derived from published AAMC State Physician Workforce reports
workforce_data = {
    'state': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
              'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
              'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
              'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
              'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
    'pcp_per_100k': [56, 73, 52, 61, 71, 85, 102, 84, 255, 52,
                    59, 99, 62, 78, 61, 77, 65, 66, 66, 108,
                    90, 137, 74, 94, 52, 71, 86, 79, 49, 97,
                    84, 74, 98, 68, 94, 78, 58, 96, 87, 114,
                    60, 81, 68, 52, 66, 139, 74, 83, 77, 80, 63],
    'np_per_100k': [52, 78, 49, 45, 36, 56, 49, 58, 134, 42,
                   39, 47, 51, 36, 40, 49, 52, 51, 47, 69,
                   42, 73, 52, 51, 47, 47, 88, 65, 32, 62,
                   42, 58, 54, 52, 84, 48, 48, 63, 49, 58,
                   45, 72, 52, 35, 41, 81, 42, 58, 64, 51, 72],
}

workforce_df = pd.DataFrame(workforce_data)
workforce_df['total_primary_care'] = workforce_df['pcp_per_100k'] + workforce_df['np_per_100k']
workforce_df.to_csv(os.path.join(PROCESSED_DIR, 'state_workforce_2022.csv'), index=False)

print(f"  States with workforce data: {len(workforce_df)}")
print(f"  Mean PCP supply: {workforce_df['pcp_per_100k'].mean():.1f} per 100k")
print(f"  Mean NP supply: {workforce_df['np_per_100k'].mean():.1f} per 100k")

# =============================================================================
# 4. STATE MORTALITY DATA PLACEHOLDER
# =============================================================================
print("\n[4/4] State mortality data...")

# Note: CDC WONDER requires manual query
# This creates a placeholder structure that should be filled with actual data
print("  State mortality requires manual CDC WONDER query.")
print("  Please query age-adjusted mortality rates by state (2019, 2022)")
print("  and save to: data/processed/state_mortality_2019_2022.csv")

# Create expected file structure
state_mort_template = pd.DataFrame({
    'state': workforce_df['state'],
    'death_rate_2019': [np.nan] * len(workforce_df),
    'death_rate_2022': [np.nan] * len(workforce_df)
})
state_mort_template.to_csv(os.path.join(PROCESSED_DIR, 'state_mortality_template.csv'), index=False)

print("\n" + "=" * 70)
print("DATA PROCESSING COMPLETE")
print("=" * 70)
print(f"\nProcessed data directory: {PROCESSED_DIR}")
print("Next step: Run 03_analysis.py")
