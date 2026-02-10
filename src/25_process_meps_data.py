
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os

# Configuration
YEARS = [2018, 2019, 2020, 2021]
BASE_URL = "https://meps.ahrq.gov/mepsweb/data_files/pufs"
DATA_DIR = "/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/raw/meps"
PROCESSED_DIR = "/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/processed/meps"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# File mapping (HC-233 is 2021 Consolidated, etc.)
# These are the standard Full Year Consolidated Data Files
FILES = {
    2021: "h233",
    2020: "h224",
    2019: "h216",
    2018: "h209"
}

def download_meps(year):
    """Download MEPS public use file if not exists."""
    file_id = FILES.get(year)
    if not file_id:
        print(f"No file ID for {year}")
        return None
        
    url = f"{BASE_URL}/{file_id}ssp.zip" # SAS transport format usually available
    local_path = f"{DATA_DIR}/{file_id}.ssp"
    
    if os.path.exists(local_path):
        print(f"File {local_path} already exists.")
        return local_path
        
    print(f"Downloading {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print(f"Downloaded and extracted {file_id}")
        return local_path
    except Exception as e:
        print(f"Failed to download {year}: {e}")
        return None

def process_meps_year(year):
    """
    Process individual year MEPS data to extract Primary Care variables.
    Key Variables of Interest:
    - HAVEUS42: Has usual source of care
    - LOCATN42: Location of usual source of care (1=Office, 2=Hosp, 3=ER)
    - APPT42: Difficulty getting appointment
    - OBVEXP{yr}: Office-based physician visits expense
    - OBTOTV{yr}: Total office-based visits
    """
    file_id = FILES.get(year)
    path = f"{DATA_DIR}/{file_id}.SSP" # SAS XPORT
    
    # Note: Requires pandas entry for sas7bdat or xport
    try:
        # Check if converted .dta or .csv exists, else read xport
        df = pd.read_sas(path, format='xport')
    except Exception:
        print(f"Could not read {path}. Ensure file exists and is XPORT format.")
        return None

    print(f"Processing {year} (N={len(df)})")
    
    # Standardize column names (MEPS changes suffixes often, e.g. 21, 20)
    # We map to common names
    
    suffix = str(year)[-2:]
    
    # Variable Mapping
    cols = {
        'DUPERSID': 'id',
        'AGELAST': 'age',
        'SEX': 'sex',
        'RACETHX': 'race_eth',
        'INSCOV': 'insurance', # Summary variable
        'POVCAT': 'poverty_cat',
        'HAVEUS42': 'has_usual_source',
        f'OBTOTV{suffix}': 'office_visits_total',
        f'OBVEXP{suffix}': 'office_visits_expenditure',
        f'TOTEXP{suffix}': 'total_expenditure'
    }
    
    # Select available columns
    available = [c for c in cols.keys() if c in df.columns]
    metrics = df[available].rename(columns=cols)
    
    metrics['year'] = year
    
    # Derived Metrics
    if 'has_usual_source' in metrics.columns:
        metrics['has_pcp'] = (metrics['has_usual_source'] == 1).astype(int)
        
    return metrics

def analyze_access_disparities(df):
    """Analyze disparities in Usual Source of Care."""
    print("\n--- Access Disparities (Has Usual Source of Care) ---")
    grp = df.groupby(['year', 'poverty_cat'])['has_pcp'].mean().unstack()
    print(grp)
    
    print("\n--- Expenditure Disparities (Mean Office Visit Exp) ---")
    grp_exp = df.groupby(['year', 'poverty_cat'])['office_visits_expenditure'].mean().unstack()
    print(grp_exp)

if __name__ == "__main__":
    print("Starting MEPS Data Pipeline...")
    
    # 1. Download (simulated structure for logic validation if offline)
    # for y in YEARS:
    #     download_meps(y)
    
    # 2. Process (Logic demonstration)
    print("pipeline ready for execution when .ssp files are present.")
    print("script extracts access (haveus42) and utilization (obtotv) metrics.")
    
    # Save schematic output structure for downstream
    schema = pd.DataFrame(columns=['id', 'year', 'age', 'sex', 'race_eth', 'has_pcp', 'office_visits_total', 'total_expenditure'])
    schema.to_csv(f"{PROCESSED_DIR}/meps_schema_definition.csv", index=False)
    print(f"Saved data schema to {PROCESSED_DIR}/meps_schema_definition.csv")

