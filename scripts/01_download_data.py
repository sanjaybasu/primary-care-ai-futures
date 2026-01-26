#!/usr/bin/env python3
"""
Download Public Data Sources for Primary Care Mortality Analysis

This script downloads publicly available data from:
1. NHANES Public-Use Linked Mortality Files (CDC/NCHS)
2. CDC WONDER State Mortality Data (manual download required)
3. CDC Social Vulnerability Index (ATSDR)

Data Sources:
- NHANES-NDI: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
- CDC WONDER: https://wonder.cdc.gov/
- CDC SVI: https://www.atsdr.cdc.gov/placeandhealth/svi/

Author: Sanjay Basu
License: MIT
"""

import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print("PRIMARY CARE MORTALITY ANALYSIS: DATA DOWNLOAD")
print("=" * 70)

# =============================================================================
# 1. NHANES PUBLIC-USE LINKED MORTALITY FILES
# =============================================================================
print("\n[1/4] Downloading NHANES Public-Use Linked Mortality Files...")

NHANES_MORTALITY_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
NHANES_CYCLES = [
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

# Download each mortality file
for filename in NHANES_CYCLES:
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        print(f"  {filename} already exists, skipping...")
        continue

    url = f"{NHANES_MORTALITY_URL}{filename}"
    try:
        print(f"  Downloading {filename}...")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"    Downloaded: {len(response.content):,} bytes")
    except Exception as e:
        print(f"    Warning: Could not download {filename}: {e}")

# =============================================================================
# 2. CDC SOCIAL VULNERABILITY INDEX
# =============================================================================
print("\n[2/4] Downloading CDC Social Vulnerability Index...")

SVI_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_US.csv"
svi_filepath = os.path.join(RAW_DIR, "SVI_2022_US.csv")

if os.path.exists(svi_filepath):
    print(f"  SVI data already exists, skipping...")
else:
    try:
        print(f"  Downloading SVI 2022 data...")
        response = requests.get(SVI_URL, timeout=300)
        response.raise_for_status()
        with open(svi_filepath, 'wb') as f:
            f.write(response.content)
        print(f"    Downloaded: {len(response.content):,} bytes")
    except Exception as e:
        print(f"    Warning: Could not download SVI data: {e}")
        print("    Please manually download from: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html")

# =============================================================================
# 3. CDC WONDER MORTALITY DATA (Manual Download Required)
# =============================================================================
print("\n[3/4] CDC WONDER Mortality Data...")
print("  CDC WONDER requires manual query. Please:")
print("  1. Go to https://wonder.cdc.gov/")
print("  2. Select 'Compressed Mortality File'")
print("  3. Query state-level age-adjusted mortality rates for 2014-2023")
print("  4. Save results to: data/raw/wonder_state_mortality.txt")

# =============================================================================
# 4. MEDICAID EXPANSION STATUS
# =============================================================================
print("\n[4/4] Creating Medicaid expansion status file...")

expansion_data = {
    'state': ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
              'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
              'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
              'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
              'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'],
    'expansion_year': [2015, None, 2014, 2014, 2014, 2014, 2014, 2014, 2014, None,
                       None, 2014, 2014, 2020, 2014, 2015, None, 2014, 2016, 2014,
                       2014, 2019, 2014, 2014, 2021, None, 2016, 2023, 2014, 2020,
                       2014, 2014, 2014, 2014, 2014, 2014, 2024, 2014, 2015, 2014,
                       None, None, None, None, 2020, 2019, 2014, 2014, None, 2014, None],
    'expanded_by_2022': [1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                         0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                         1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                         1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                         0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
}

expansion_df = pd.DataFrame(expansion_data)
expansion_df.to_csv(os.path.join(PROCESSED_DIR, 'medicaid_expansion_status.csv'), index=False)
print(f"  Saved Medicaid expansion data for {len(expansion_df)} states")
print(f"    Expansion states (by 2022): {expansion_df['expanded_by_2022'].sum()}")
print(f"    Non-expansion states: {len(expansion_df) - expansion_df['expanded_by_2022'].sum()}")

print("\n" + "=" * 70)
print("DATA DOWNLOAD COMPLETE")
print("=" * 70)
print(f"\nData directory: {RAW_DIR}")
print("Next step: Run 02_process_data.py")
