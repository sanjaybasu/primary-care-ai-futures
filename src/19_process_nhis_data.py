#!/usr/bin/env python3
"""
Process NHIS Data Files (2018-2022)

Extracts key primary care access, utilization, and barrier measures from 
National Health Interview Survey data.

Data: https://www.cdc.gov/nchs/nhis/data-questionnaires-documentation.htm
"""

import pandas as pd
import zipfile
from pathlib import Path
import io

DATA_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/raw")
OUTPUT_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NHIS 2018 uses the old questionnaire format (samadult)
# NHIS 2019+ uses the redesigned questionnaire (adult)

# Key variables for primary care access analysis
# Variable positions come from NHIS documentation

# 2018 SAMADULT variables (old questionnaire)
NHIS_2018_VARS = {
    # Access measures
    'AUSESSION': {'start': 1, 'end': 6, 'description': 'Record ID'},
    'AHESSION': {'start': 7, 'end': 12, 'description': 'Household serial number'},
    'USUALPL': {'start': 485, 'end': 486, 'description': 'Place usually go when sick'},
    'USPLKND': {'start': 487, 'end': 488, 'description': 'Kind of place usually go'},
    'AHCNOFUL': {'start': 513, 'end': 514, 'description': 'Couldnt get care - cost'},
    'AHCNORED': {'start': 517, 'end': 518, 'description': 'Couldnt get care - past due'},
    'AHCNONE': {'start': 519, 'end': 520, 'description': 'Did NOT need health care past 12m'},
    # Utilization
    'AHCDLYR1': {'start': 493, 'end': 494, 'description': 'Delayed care - cost'},
    'AHCDLYR2': {'start': 495, 'end': 496, 'description': 'Delayed care - no transportation'},
    'AHCDLYR3': {'start': 497, 'end': 498, 'description': 'Delayed care - couldnt get appt'},
    'AHCDLYR4': {'start': 499, 'end': 500, 'description': 'Delayed care - too long wait'},
    'AHCDLYR5': {'start': 501, 'end': 502, 'description': 'Delayed care - not open'},
    # Demographics
    'SEX': {'start': 31, 'end': 32, 'description': 'Sex'},
    'AGE_P': {'start': 33, 'end': 35, 'description': 'Age'},
    'RACERPI2': {'start': 39, 'end': 40, 'description': 'Race'},
    'HISCODI3': {'start': 41, 'end': 42, 'description': 'Hispanic origin'},
    # Insurance
    'NOTCOV': {'start': 141, 'end': 142, 'description': 'Not covered by health insurance'},
    'MEDICARE': {'start': 143, 'end': 144, 'description': 'Medicare coverage'},
    'MEDICAID': {'start': 147, 'end': 148, 'description': 'Medicaid coverage'},
    'PRIVATE': {'start': 145, 'end': 146, 'description': 'Private coverage'},
    # Weight
    'WTFA_SA': {'start': 611, 'end': 621, 'description': 'Sample adult weight'},
}

# 2020-2022 use the redesigned questionnaire with different variable names
NHIS_2020_PLUS_VARS = {
    # These files are actually CSV format in the newer years, need to check
    # Variable names from adult questionnaire documentation
    'URESSION': 'Record ID',
    'USUALPL_A': 'Usual place for health care',
    'URGNT12MTC_A': 'Urgent care past 12m',
    'MEDDL_A': 'Delayed medical care due to cost',
    'MEDNG_A': 'Needed but didnt get medical care',
    'PAYWORRY_A': 'Worry about paying medical bills',
    'SEX_A': 'Sex',
    'AGEP_A': 'Age',
    'RACEALLP_A': 'Race',
    'HISPALLP_A': 'Hispanic origin',
    'NOTCOV_A': 'Not covered',
    'WTFA_A': 'Final weight',
}

def parse_fixed_width_2018(zip_path):
    """Parse 2018 NHIS fixed-width file."""
    print(f"Processing {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            content = f.read().decode('latin-1')
    
    lines = content.strip().split('\n')
    print(f"  Total records: {len(lines)}")
    
    # Parse key variables
    data = []
    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print(f"  Processing record {i}...")
        
        try:
            record = {}
            for var_name, var_info in NHIS_2018_VARS.items():
                start = var_info['start'] - 1  # Convert to 0-indexed
                end = var_info['end']
                value = line[start:end].strip()
                record[var_name] = value if value else None
            data.append(record)
        except Exception as e:
            continue
    
    df = pd.DataFrame(data)
    print(f"  Parsed {len(df)} records")
    return df

def parse_newer_nhis(zip_path, year):
    """Parse 2020+ NHIS files (check format first)."""
    print(f"Processing {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            # Read first few bytes to detect format
            header = f.read(1000).decode('latin-1')
            
    # Check if it's CSV or fixed-width
    if ',' in header[:100]:
        print("  Detected CSV format")
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(filename) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        print("  Detected fixed-width format")
        # Parse similar to 2018 but with different positions
        # For now, return empty - would need specific parsing specs
        df = pd.DataFrame()
    
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df

def compute_access_measures(df_2018):
    """Compute summary access measures from 2018 data."""
    if df_2018.empty:
        return {}
    
    # Convert to numeric where possible
    for col in df_2018.columns:
        df_2018[col] = pd.to_numeric(df_2018[col], errors='coerce')
    
    # Usual source of care (1=Yes clinic, 2=Yes but ER, 3=No)
    # 1 or 2 = has usual source
    df_2018['has_usual_source'] = df_2018['USUALPL'].isin([1, 2])
    
    # Delayed care due to cost
    df_2018['delayed_cost'] = df_2018['AHCDLYR1'] == 1
    
    # Any barrier to care (transportation, appointment, wait, hours)
    barrier_cols = ['AHCDLYR2', 'AHCDLYR3', 'AHCDLYR4', 'AHCDLYR5']
    df_2018['any_access_barrier'] = df_2018[barrier_cols].apply(
        lambda row: any(row == 1), axis=1
    )
    
    # Uninsured
    df_2018['uninsured'] = df_2018['NOTCOV'] == 1
    
    # Medicaid
    df_2018['has_medicaid'] = df_2018['MEDICAID'] == 1
    
    # Apply weights for population estimates
    if 'WTFA_SA' in df_2018.columns:
        total_weight = df_2018['WTFA_SA'].sum()
        
        measures = {
            'year': 2018,
            'n_respondents': len(df_2018),
            'total_weight': total_weight,
            'pct_usual_source': (df_2018['has_usual_source'] * df_2018['WTFA_SA']).sum() / total_weight * 100,
            'pct_delayed_cost': (df_2018['delayed_cost'] * df_2018['WTFA_SA']).sum() / total_weight * 100,
            'pct_any_barrier': (df_2018['any_access_barrier'] * df_2018['WTFA_SA']).sum() / total_weight * 100,
            'pct_uninsured': (df_2018['uninsured'] * df_2018['WTFA_SA']).sum() / total_weight * 100,
            'pct_medicaid': (df_2018['has_medicaid'] * df_2018['WTFA_SA']).sum() / total_weight * 100,
        }
    else:
        measures = {
            'year': 2018,
            'n_respondents': len(df_2018),
            'pct_usual_source': df_2018['has_usual_source'].mean() * 100,
            'pct_delayed_cost': df_2018['delayed_cost'].mean() * 100,
            'pct_any_barrier': df_2018['any_access_barrier'].mean() * 100,
            'pct_uninsured': df_2018['uninsured'].mean() * 100,
            'pct_medicaid': df_2018['has_medicaid'].mean() * 100,
        }
    
    return measures

def main():
    print("=" * 60)
    print("NHIS DATA PROCESSING")
    print("=" * 60)
    
    all_measures = []
    
    # Process 2018
    nhis_2018_path = DATA_DIR / "nhis_2018.zip"
    if nhis_2018_path.exists():
        df_2018 = parse_fixed_width_2018(nhis_2018_path)
        measures = compute_access_measures(df_2018)
        all_measures.append(measures)
        
        # Save processed data
        output_path = OUTPUT_DIR / "nhis_2018_access.csv"
        df_2018.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    # Process 2020-2022
    for year in [2020, 2021, 2022]:
        zip_path = DATA_DIR / f"nhis_{year}.zip"
        if zip_path.exists():
            df = parse_newer_nhis(zip_path, year)
            if not df.empty:
                output_path = OUTPUT_DIR / f"nhis_{year}_access.csv"
                # Select relevant columns if they exist
                relevant_cols = [c for c in df.columns if any(
                    kw in c.upper() for kw in ['USUAL', 'DELAY', 'MEDD', 'MEDNG', 'NOTCOV', 'WTFA', 'SEX', 'AGE', 'RACE', 'HISP']
                )]
                if relevant_cols:
                    df[relevant_cols].to_csv(output_path, index=False)
                    print(f"Saved: {output_path}")
    
    # Summary
    if all_measures:
        summary_df = pd.DataFrame(all_measures)
        summary_path = OUTPUT_DIR / "nhis_access_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        print("\n" + summary_df.to_string())

if __name__ == "__main__":
    main()
