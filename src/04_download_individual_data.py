#!/usr/bin/env python3
"""
Download individual-level public use data for causal inference analysis.

KEY DATASETS WITH INDIVIDUAL-LEVEL DATA:
1. NHANES + Public-Use Linked Mortality Files (NCHS releases these publicly!)
2. MEPS Public Use Files (AHRQ)
3. NHIS Public Use Files (CDC/NCHS)

NHANES-NDI PUBLIC LINKAGE is the key - it provides individual mortality follow-up!
"""

import os
import requests
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INDIVIDUAL_DIR = os.path.join(DATA_DIR, 'individual')

for d in [RAW_DIR, INDIVIDUAL_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("DOWNLOADING INDIVIDUAL-LEVEL PUBLIC USE DATA")
print("=" * 70)

# ============================================================================
# 1. NHANES PUBLIC-USE LINKED MORTALITY FILES
# ============================================================================
print("\n[1/4] NHANES Public-Use Linked Mortality Files...")
print("      Source: NCHS - these include mortality follow-up through 2019!")

# NHANES Linked Mortality Files are publicly available
# https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
# Files available for NHANES III (1988-1994) and continuous NHANES (1999-2018)

NHANES_MORTALITY_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"

# Download NHANES 2017-2018 mortality linkage (most recent with mortality follow-up)
nhanes_files = {
    'NHANES_2017_2018_MORT': 'NHANES_2017_2018_MORT_2019_PUBLIC.dat',
    'NHANES_2015_2016_MORT': 'NHANES_2015_2016_MORT_2019_PUBLIC.dat',
    'NHANES_2013_2014_MORT': 'NHANES_2013_2014_MORT_2019_PUBLIC.dat',
    'NHANES_2011_2012_MORT': 'NHANES_2011_2012_MORT_2019_PUBLIC.dat',
    'NHANES_2009_2010_MORT': 'NHANES_2009_2010_MORT_2019_PUBLIC.dat',
    'NHANES_2007_2008_MORT': 'NHANES_2007_2008_MORT_2019_PUBLIC.dat',
    'NHANES_2005_2006_MORT': 'NHANES_2005_2006_MORT_2019_PUBLIC.dat',
    'NHANES_2003_2004_MORT': 'NHANES_2003_2004_MORT_2019_PUBLIC.dat',
    'NHANES_2001_2002_MORT': 'NHANES_2001_2002_MORT_2019_PUBLIC.dat',
    'NHANES_1999_2000_MORT': 'NHANES_1999_2000_MORT_2019_PUBLIC.dat',
}

# Column specifications for the mortality file (fixed-width)
# From NCHS documentation
mort_colspecs = [
    (0, 14),    # SEQN - Respondent sequence number
    (14, 15),   # ELIGSTAT - Eligibility Status for Mortality Follow-up
    (15, 16),   # MORTSTAT - Final Mortality Status
    (16, 19),   # UCOD_LEADING - Underlying Cause of Death: Recode
    (19, 22),   # DIABETES - Diabetes flag from multiple cause of death
    (22, 25),   # HYPERTEN - Hypertension flag from multiple cause of death
    (25, 28),   # DODQTR - Quarter of death (if applicable)
    (28, 32),   # DODYEAR - Year of death (if applicable)
    (32, 34),   # WGT_NEW - Weight for mortality follow-up
    (34, 43),   # SA_WGT_NEW - Weight adjusted for hepatitis hepatic steatosis sub-sample
]

mort_names = ['SEQN', 'ELIGSTAT', 'MORTSTAT', 'UCOD_LEADING', 'DIABETES',
              'HYPERTEN', 'DODQTR', 'DODYEAR', 'WGT_NEW', 'SA_WGT_NEW']

all_mortality_data = []

for name, filename in nhanes_files.items():
    url = NHANES_MORTALITY_URL + filename
    print(f"    Downloading {name}...")
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            # Parse fixed-width file
            from io import StringIO
            df = pd.read_fwf(StringIO(response.text), colspecs=mort_colspecs,
                            names=mort_names, dtype=str)
            df['survey_cycle'] = name.replace('NHANES_', '').replace('_MORT', '')
            all_mortality_data.append(df)
            print(f"      Loaded {len(df)} records")
        else:
            print(f"      Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"      Error: {e}")

if all_mortality_data:
    nhanes_mortality = pd.concat(all_mortality_data, ignore_index=True)
    nhanes_mortality.to_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_mortality_linked.csv'), index=False)
    print(f"\n    TOTAL NHANES mortality records: {len(nhanes_mortality)}")

    # Convert to numeric
    nhanes_mortality['MORTSTAT'] = pd.to_numeric(nhanes_mortality['MORTSTAT'], errors='coerce')
    nhanes_mortality['ELIGSTAT'] = pd.to_numeric(nhanes_mortality['ELIGSTAT'], errors='coerce')

    eligible = nhanes_mortality[nhanes_mortality['ELIGSTAT'] == 1]
    deaths = eligible[eligible['MORTSTAT'] == 1]
    print(f"    Eligible for follow-up: {len(eligible)}")
    print(f"    Deaths observed: {len(deaths)}")
    print(f"    Mortality rate: {len(deaths)/len(eligible)*100:.2f}%")

# ============================================================================
# 2. DOWNLOAD NHANES DEMOGRAPHIC AND HEALTH DATA
# ============================================================================
print("\n[2/4] NHANES Demographic and Health Data...")

# NHANES data files are available as XPT (SAS transport) files
# We'll download key variables: demographics, insurance, healthcare access, chronic conditions

NHANES_DATA_URL = "https://wwwn.cdc.gov/Nchs/Nhanes/"

# Key files for each cycle
nhanes_data_files = {
    '2017-2018': {
        'DEMO': 'DEMO_J.XPT',
        'HIQ': 'HIQ_J.XPT',      # Health insurance
        'HUQ': 'HUQ_J.XPT',      # Hospital utilization
        'BPX': 'BPX_J.XPT',      # Blood pressure
        'BMX': 'BMX_J.XPT',      # Body measures
        'DIQ': 'DIQ_J.XPT',      # Diabetes
    },
    '2015-2016': {
        'DEMO': 'DEMO_I.XPT',
        'HIQ': 'HIQ_I.XPT',
        'HUQ': 'HUQ_I.XPT',
        'BPX': 'BPX_I.XPT',
        'BMX': 'BMX_I.XPT',
        'DIQ': 'DIQ_I.XPT',
    },
    '2013-2014': {
        'DEMO': 'DEMO_H.XPT',
        'HIQ': 'HIQ_H.XPT',
        'HUQ': 'HUQ_H.XPT',
        'BPX': 'BPX_H.XPT',
        'BMX': 'BMX_H.XPT',
        'DIQ': 'DIQ_H.XPT',
    },
    '2011-2012': {
        'DEMO': 'DEMO_G.XPT',
        'HIQ': 'HIQ_G.XPT',
        'HUQ': 'HUQ_G.XPT',
        'BPX': 'BPX_G.XPT',
        'BMX': 'BMX_G.XPT',
        'DIQ': 'DIQ_G.XPT',
    },
}

def download_xpt(url):
    """Download and read XPT file."""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return pd.read_sas(BytesIO(response.content), format='xport')
    except Exception as e:
        print(f"      Error: {e}")
    return None

all_nhanes_data = []

for cycle, files in nhanes_data_files.items():
    print(f"    Downloading NHANES {cycle}...")
    cycle_data = None

    for file_type, filename in files.items():
        url = f"{NHANES_DATA_URL}{cycle}/{filename}"
        df = download_xpt(url)
        if df is not None:
            if cycle_data is None:
                cycle_data = df
            else:
                # Merge on SEQN
                cycle_data = cycle_data.merge(df, on='SEQN', how='outer', suffixes=('', f'_{file_type}'))
            print(f"      {file_type}: {len(df)} records")

    if cycle_data is not None:
        cycle_data['survey_cycle'] = cycle
        all_nhanes_data.append(cycle_data)

if all_nhanes_data:
    nhanes_full = pd.concat(all_nhanes_data, ignore_index=True)
    print(f"\n    Total NHANES records (with health data): {len(nhanes_full)}")

    # Save
    nhanes_full.to_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_health_data.csv'), index=False)

# ============================================================================
# 3. DOWNLOAD MEPS PUBLIC USE FILES
# ============================================================================
print("\n[3/4] MEPS Full Year Consolidated Files...")
print("      Source: AHRQ Medical Expenditure Panel Survey")

# MEPS data available from AHRQ
MEPS_BASE_URL = "https://meps.ahrq.gov/mepsweb/data_files/pufs/"

# Full year consolidated files
meps_files = {
    '2022': 'h233.zip',  # Full Year Consolidated 2022
    '2021': 'h224.zip',  # Full Year Consolidated 2021
    '2020': 'h216.zip',  # Full Year Consolidated 2020
    '2019': 'h209.zip',  # Full Year Consolidated 2019
    '2018': 'h201.zip',  # Full Year Consolidated 2018
}

for year, filename in meps_files.items():
    url = f"{MEPS_BASE_URL}{filename}"
    print(f"    Downloading MEPS {year} ({filename})...")
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            filepath = os.path.join(RAW_DIR, f'meps_{year}.zip')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"      Downloaded: {len(response.content)/1024/1024:.1f} MB")

            # Extract and read
            try:
                with zipfile.ZipFile(filepath, 'r') as z:
                    # Find the data file (usually .ssp or .dat)
                    for name in z.namelist():
                        if name.endswith('.ssp') or name.endswith('.dat'):
                            print(f"      Contains: {name}")
            except:
                pass
        else:
            print(f"      Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"      Error: {e}")

# ============================================================================
# 4. DOWNLOAD NHIS PUBLIC USE FILES
# ============================================================================
print("\n[4/4] NHIS Public Use Files...")
print("      Source: CDC/NCHS National Health Interview Survey")

# NHIS data available from CDC
NHIS_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/"

# Sample Adult files (most relevant for mortality analysis)
nhis_files = {
    '2022': '2022/adult22.zip',
    '2021': '2021/adult21.zip',
    '2020': '2020/adult20.zip',
    '2019': '2019/samadult.zip',
    '2018': '2018/samadult.zip',
}

for year, filepath in nhis_files.items():
    url = f"{NHIS_BASE_URL}{filepath}"
    print(f"    Downloading NHIS {year}...")
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            outpath = os.path.join(RAW_DIR, f'nhis_{year}.zip')
            with open(outpath, 'wb') as f:
                f.write(response.content)
            print(f"      Downloaded: {len(response.content)/1024/1024:.1f} MB")
        else:
            print(f"      Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"      Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

print("\nFiles in individual data directory:")
for f in os.listdir(INDIVIDUAL_DIR):
    fpath = os.path.join(INDIVIDUAL_DIR, f)
    size = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")

print("\nFiles in raw data directory:")
for f in sorted(os.listdir(RAW_DIR)):
    fpath = os.path.join(RAW_DIR, f)
    size = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")

print("""
NEXT STEPS:
1. Merge NHANES health data with mortality linkage files
2. Process MEPS files for healthcare utilization analysis
3. Process NHIS files for access and health status
4. Link individual data to county characteristics via geographic codes
""")
