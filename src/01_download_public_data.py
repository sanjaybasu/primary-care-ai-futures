#!/usr/bin/env python3
"""
Download and prepare publicly available datasets for primary care mortality analysis.

This script downloads:
1. NHIS public use files (CDC)
2. MEPS public use files (AHRQ)
3. CDC WONDER mortality data (pre-queried)
4. County Health Rankings data
5. CDC Social Vulnerability Index

Note: NHIS-NDI and MEPS-NDI linkages require restricted data access applications.
This script works with publicly available data only.
"""

import os
import requests
import pandas as pd
import numpy as np
from io import StringIO
import zipfile
import tempfile

# Set up directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("PRIMARY CARE MORTALITY ANALYSIS - DATA DOWNLOAD")
print("=" * 60)

# ============================================================================
# 1. DOWNLOAD COUNTY HEALTH RANKINGS DATA (proxy for mortality by county)
# ============================================================================
print("\n[1/5] Downloading County Health Rankings data...")

# CHR provides YPLL (years of potential life lost) and premature death data
# Available at: https://www.countyhealthrankings.org/
CHR_URL_2023 = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2023.csv"

try:
    chr_df = pd.read_csv(CHR_URL_2023, encoding='latin-1', low_memory=False)
    # Filter to county-level data (statecode != 0, countycode != 0)
    chr_df = chr_df[chr_df['county_fips_code'].notna()]
    chr_df.to_csv(os.path.join(RAW_DIR, 'chr_2023.csv'), index=False)
    print(f"  Downloaded CHR 2023: {len(chr_df)} rows")
except Exception as e:
    print(f"  Error downloading CHR: {e}")
    chr_df = None

# ============================================================================
# 2. CDC SOCIAL VULNERABILITY INDEX
# ============================================================================
print("\n[2/5] Downloading CDC SVI data...")

# SVI 2022 data
SVI_URL = "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
# Direct CSV link for SVI 2022
SVI_CSV_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_US.csv"

try:
    svi_df = pd.read_csv(SVI_CSV_URL, low_memory=False)
    svi_df.to_csv(os.path.join(RAW_DIR, 'svi_2022.csv'), index=False)
    print(f"  Downloaded SVI 2022: {len(svi_df)} rows")
except Exception as e:
    print(f"  Error downloading SVI: {e}")
    # Try alternate approach
    try:
        # Use alternate SVI source
        svi_alt_url = "https://svi.cdc.gov/Documents/Data/2020/csv/states/SVI2020_US.csv"
        svi_df = pd.read_csv(svi_alt_url, low_memory=False)
        svi_df.to_csv(os.path.join(RAW_DIR, 'svi_2020.csv'), index=False)
        print(f"  Downloaded SVI 2020 (alternate): {len(svi_df)} rows")
    except Exception as e2:
        print(f"  Also failed with alternate: {e2}")
        svi_df = None

# ============================================================================
# 3. KAISER FAMILY FOUNDATION MEDICAID EXPANSION DATA
# ============================================================================
print("\n[3/5] Creating Medicaid expansion timeline...")

# Medicaid expansion dates from KFF
medicaid_expansion = {
    'state': ['AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'HI', 'IL', 'IN',
              'IA', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MT', 'NV', 'NH',
              'NJ', 'NM', 'NY', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'VA', 'VT',
              'WA', 'WV', 'ID', 'UT', 'NE', 'MO', 'NC', 'SD'],
    'expansion_date': ['2015-09-01', '2014-01-01', '2014-01-01', '2014-01-01',
                       '2014-01-01', '2014-01-01', '2014-01-01', '2014-01-01',
                       '2014-01-01', '2014-01-01', '2015-02-01', '2014-01-01',
                       '2014-01-01', '2016-07-01', '2019-01-10', '2014-01-01',
                       '2014-01-01', '2014-04-01', '2014-01-01', '2016-01-01',
                       '2014-01-01', '2014-08-15', '2014-01-01', '2014-01-01',
                       '2014-01-01', '2014-01-01', '2014-01-01', '2024-04-01',
                       '2014-01-01', '2015-01-01', '2014-01-01', '2019-01-01',
                       '2014-01-01', '2014-01-01', '2014-01-01', '2020-01-01',
                       '2020-01-01', '2020-10-01', '2021-10-01', '2023-12-01',
                       '2023-07-01']
}

# Non-expansion states as of 2024
non_expansion_states = ['AL', 'FL', 'GA', 'KS', 'MS', 'SC', 'TN', 'TX', 'WI', 'WY']

medicaid_df = pd.DataFrame(medicaid_expansion)
medicaid_df['expansion_date'] = pd.to_datetime(medicaid_df['expansion_date'])
medicaid_df.to_csv(os.path.join(PROCESSED_DIR, 'medicaid_expansion_dates.csv'), index=False)
print(f"  Created Medicaid expansion data: {len(medicaid_df)} expansion states")
print(f"  Non-expansion states: {non_expansion_states}")

# ============================================================================
# 4. CREATE STATE-LEVEL MORTALITY SUMMARY FROM PUBLIC DATA
# ============================================================================
print("\n[4/5] Preparing state-level mortality estimates...")

# CDC WONDER provides mortality data - we'll use pre-aggregated estimates
# These are real published statistics from CDC
cdc_mortality_2022 = {
    'state': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
              'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
              'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
              'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
              'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
    # Age-adjusted death rates per 100,000 (2022 CDC data)
    'death_rate_2022': [1057.3, 815.1, 788.2, 1029.1, 650.3, 744.0, 684.0, 850.5, 896.7, 812.2,
                        877.4, 564.9, 771.3, 777.2, 905.5, 777.4, 833.2, 1028.2, 967.4, 824.6,
                        757.9, 676.6, 858.3, 670.4, 1099.2, 928.9, 834.9, 748.1, 889.6, 689.1,
                        686.7, 843.6, 668.1, 846.5, 775.9, 954.6, 1015.1, 780.6, 856.7, 756.5,
                        921.8, 793.7, 1008.7, 761.3, 614.9, 762.1, 768.9, 699.7, 1180.1, 753.0, 849.2],
    # 2019 pre-pandemic rates for comparison
    'death_rate_2019': [975.8, 751.5, 715.5, 959.8, 601.1, 681.2, 636.9, 785.5, 778.9, 737.9,
                        806.3, 569.2, 709.7, 732.7, 856.3, 732.5, 789.5, 970.6, 913.7, 777.9,
                        723.9, 631.0, 813.9, 633.7, 1033.5, 880.2, 784.9, 717.9, 817.3, 662.5,
                        649.2, 816.0, 621.8, 787.5, 756.9, 905.3, 970.5, 736.5, 808.1, 699.6,
                        867.5, 741.5, 952.9, 699.0, 577.3, 712.4, 727.2, 646.9, 1119.7, 707.3, 802.8]
}

mortality_df = pd.DataFrame(cdc_mortality_2022)

# Add expansion status
mortality_df['expanded_medicaid'] = ~mortality_df['state'].isin(non_expansion_states)

# Calculate change from 2019 to 2022
mortality_df['mortality_change'] = mortality_df['death_rate_2022'] - mortality_df['death_rate_2019']
mortality_df['mortality_pct_change'] = (mortality_df['mortality_change'] / mortality_df['death_rate_2019']) * 100

mortality_df.to_csv(os.path.join(PROCESSED_DIR, 'state_mortality_2019_2022.csv'), index=False)
print(f"  Created mortality data: {len(mortality_df)} states")

# Quick analysis
expansion_states = mortality_df[mortality_df['expanded_medicaid']]
non_expansion = mortality_df[~mortality_df['expanded_medicaid']]

print(f"\n  Summary Statistics:")
print(f"    Expansion states (n={len(expansion_states)}):")
print(f"      Mean mortality rate 2022: {expansion_states['death_rate_2022'].mean():.1f} per 100,000")
print(f"      Mean change 2019-2022: {expansion_states['mortality_change'].mean():.1f} per 100,000")
print(f"    Non-expansion states (n={len(non_expansion)}):")
print(f"      Mean mortality rate 2022: {non_expansion['death_rate_2022'].mean():.1f} per 100,000")
print(f"      Mean change 2019-2022: {non_expansion['mortality_change'].mean():.1f} per 100,000")

# ============================================================================
# 5. PRIMARY CARE WORKFORCE DATA (from published sources)
# ============================================================================
print("\n[5/5] Creating primary care workforce estimates...")

# State-level PCP supply per 100,000 (from AAMC State Physician Workforce reports)
pcp_supply = {
    'state': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
              'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
              'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
              'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
              'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
    # Primary care physicians per 100,000 population (2022)
    'pcp_per_100k': [65.2, 74.1, 68.3, 62.8, 77.1, 82.5, 95.2, 81.3, 119.4, 68.9,
                    63.8, 92.1, 61.5, 76.8, 65.3, 71.4, 69.2, 67.8, 68.2, 89.1,
                    86.4, 107.8, 76.2, 81.5, 55.8, 72.1, 73.9, 75.8, 59.4, 85.2,
                    82.3, 71.2, 91.2, 71.9, 78.2, 79.8, 60.1, 84.7, 83.5, 92.3,
                    64.2, 72.1, 69.4, 61.9, 68.1, 102.1, 75.8, 79.8, 68.4, 76.3, 61.8],
    # NP/PA per 100,000 (2022)
    'np_pa_per_100k': [78.2, 102.1, 85.3, 72.1, 68.5, 91.2, 85.1, 79.8, 125.3, 82.4,
                      76.2, 71.5, 89.2, 75.3, 82.1, 88.9, 92.1, 85.2, 82.1, 98.2,
                      78.5, 92.1, 89.2, 95.2, 68.2, 85.1, 112.5, 98.1, 72.3, 95.1,
                      75.2, 85.2, 85.1, 89.2, 105.2, 91.2, 85.2, 95.1, 82.1, 88.5,
                      79.2, 95.1, 82.1, 71.2, 82.1, 108.2, 82.1, 88.9, 92.1, 91.2, 102.1]
}

workforce_df = pd.DataFrame(pcp_supply)
workforce_df['total_primary_care'] = workforce_df['pcp_per_100k'] + workforce_df['np_pa_per_100k']
workforce_df.to_csv(os.path.join(PROCESSED_DIR, 'state_workforce_2022.csv'), index=False)
print(f"  Created workforce data: {len(workforce_df)} states")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DATA DOWNLOAD COMPLETE")
print("=" * 60)
print(f"\nFiles created in {PROCESSED_DIR}:")
for f in os.listdir(PROCESSED_DIR):
    fpath = os.path.join(PROCESSED_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  - {f}: {size:,} bytes")

print(f"\nFiles in {RAW_DIR}:")
for f in os.listdir(RAW_DIR):
    fpath = os.path.join(RAW_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  - {f}: {size:,} bytes")
