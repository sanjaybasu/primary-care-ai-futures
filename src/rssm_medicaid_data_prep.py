"""
RSSM Medicaid Data Preparation
Parses 2014-2015 MEPS data for Medicaid Expansion Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path

def parse_meps_fixed_width(filepath, colspecs, names, year):
    """Parse MEPS fixed width file"""
    print(f"Parsing {filepath} ({year})...")
    
    # Read in chunks to handle large files
    chunks = []
    chunk_size = 10000
    
    try:
        for chunk in pd.read_fwf(filepath, colspecs=colspecs, names=names, chunksize=chunk_size):
            # Filter for valid rows if needed
            chunks.append(chunk)
            print(f"  Processed {len(chunk)} rows...", end='\r')
            
        df = pd.concat(chunks, ignore_index=True)
        print(f"\n  Total rows: {len(df):,}")
        
        # Add year
        df['year'] = year
        
        return df
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def prepare_medicaid_data():
    # Assuming running from healthcare_world_model directory
    data_dir = Path("data/real_meps")
    if not data_dir.exists():
        # Fallback if running from root
        data_dir = Path("healthcare_world_model/data/real_meps")
    
    # 2014 Specs (h171)
    # DUPERSID 9-16 (0-indexed: 8-16)
    # REGION14 79-80 (78-80)
    # AGE14X 177-178 (176-178)
    # ERTOT14 4188-4189 (4187-4189)
    # MCDJA14 1661-1662 (1660-1662) - Start of Medicaid monthly
    # ...
    # MCDDE14 1683-1684 (1682-1684)
    
    # Note: pd.read_fwf uses 0-indexed [start, end)
    # MEPS uses 1-indexed inclusive
    # So MEPS 9-16 -> Python 8-16
    
    specs_2014 = [
        (8, 16),      # DUPERSID
        (78, 80),     # REGION14
        (176, 178),   # AGE14X
        (4187, 4189), # ERTOT14
        # Medicaid Monthly (Jan, Feb... Dec)
        (1660, 1662), (1662, 1664), (1664, 1666), (1666, 1668),
        (1668, 1670), (1670, 1672), (1672, 1674), (1674, 1676),
        (1676, 1678), (1678, 1680), (1680, 1682), (1682, 1684)
    ]
    
    names_2014 = [
        'DUPERSID', 'REGION', 'AGE', 'ERTOT',
        'MCDJA', 'MCDFE', 'MCDMA', 'MCDAP', 'MCDMY', 'MCDJU',
        'MCDJL', 'MCDAU', 'MCDSE', 'MCDOC', 'MCDNO', 'MCDDE'
    ]
    
    # 2015 Specs (h181)
    # DUPERSID 9-16
    # REGION15 79-80
    # AGE15X 177-178
    # ERTOT15 4134-4135
    # MCDJA15 1639-1640
    # ...
    # MCDDE15 1661-1662
    
    specs_2015 = [
        (8, 16),      # DUPERSID
        (78, 80),     # REGION15
        (176, 178),   # AGE15X
        (4133, 4135), # ERTOT15
        # Medicaid Monthly
        (1638, 1640), (1640, 1642), (1642, 1644), (1644, 1646),
        (1646, 1648), (1648, 1650), (1650, 1652), (1652, 1654),
        (1654, 1656), (1656, 1658), (1658, 1660), (1660, 1662)
    ]
    
    names_2015 = names_2014 # Same names
    
    # Parse files
    df_2014 = parse_meps_fixed_width(data_dir / "h171.dat", specs_2014, names_2014, 2014)
    df_2015 = parse_meps_fixed_width(data_dir / "h181.dat", specs_2015, names_2015, 2015)
    
    if df_2014 is None or df_2015 is None:
        print("Error: Could not parse all files.")
        return
    
    # Combine
    combined = pd.concat([df_2014, df_2015], ignore_index=True)
    
    # Process Variables
    
    # Medicaid Coverage (Any month)
    mcd_cols = ['MCDJA', 'MCDFE', 'MCDMA', 'MCDAP', 'MCDMY', 'MCDJU',
                'MCDJL', 'MCDAU', 'MCDSE', 'MCDOC', 'MCDNO', 'MCDDE']
    
    # MEPS coding: 1=Yes, 2=No, -1=Inapplicable, etc.
    # We want 1=Yes, 0=No
    for col in mcd_cols:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    combined['medicaid_any'] = (combined[mcd_cols] == 1).any(axis=1).astype(int)
    
    # Clean other vars
    combined['age'] = pd.to_numeric(combined['AGE'], errors='coerce')
    combined['ed_visits'] = pd.to_numeric(combined['ERTOT'], errors='coerce').fillna(0)
    combined['region'] = pd.to_numeric(combined['REGION'], errors='coerce')
    
    # Expansion State Logic (Simplified for validation)
    # Expansion states in 2014: AZ, AR, CA, CO, CT, DE, DC, HI, IL, IA, KY, MD, MA, MN, NV, NJ, NM, NY, ND, OH, OR, RI, VT, WA, WV
    # MEPS Region: 1=Northeast, 2=Midwest, 3=South, 4=West
    # This is coarse. We really need State.
    # MEPS Public Use Files DO NOT have State identifiers (masked).
    # They only have Region and MSA.
    # This is a limitation.
    # However, we can use Region as a proxy for "High Expansion" vs "Low Expansion"?
    # Northeast (1) and West (4) had high expansion.
    # South (3) had low expansion (TX, FL, GA didn't expand).
    # Midwest (2) was mixed.
    
    # We will compare Region 1+4 (High Expansion) vs Region 3 (Low Expansion).
    combined['expansion_group'] = combined['region'].map({
        1: 'High', 4: 'High',
        3: 'Low',
        2: 'Mixed'
    })
    
    # Select columns
    final_df = combined[['DUPERSID', 'year', 'age', 'ed_visits', 'medicaid_any', 'region', 'expansion_group']]
    final_df.rename(columns={'DUPERSID': 'person_id'}, inplace=True)
    
    # Save
    output_file = "healthcare_world_model/rssm_medicaid_prepared.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved prepared data to {output_file}")
    print(final_df.groupby(['year', 'expansion_group'])['medicaid_any'].mean())

if __name__ == "__main__":
    prepare_medicaid_data()
