"""
Simplified RSSM Data Loader
Loads MEPS data directly and prepares for RSSM training
Works with existing MEPS .dta files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat
from typing import Tuple
import traceback

def load_meps_for_rssm(meps_dir: str = "healthcare_world_model/data/real_meps") -> pd.DataFrame:
    """
    Load MEPS data for RSSM training
    Returns unified dataframe with temporal features
    """
    print("Loading MEPS data for RSSM...")
    
    meps_path = Path(meps_dir)
    
    # Load all years
    years_data = []
    
    for year, filename in [(2019, 'h216.dta'), (2020, 'h224.dta'), 
                           (2021, 'h233.dta'), (2022, 'h243.dta')]:
        try:
            filepath = meps_path / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found")
                continue
            
            print(f"  Loading {year}...")
            df, meta = pyreadstat.read_dta(str(filepath))
        
            # Extract features
            year_str = str(year)[-2:]
            
            def get_col(col_name, default_val):
                if col_name in df.columns:
                    return pd.to_numeric(df[col_name], errors='coerce').fillna(default_val)
                return pd.Series([default_val] * len(df))

            data = pd.DataFrame({
                'person_id': df['DUPERSID'],
                'year': year,
                'panel': df['PANEL'],
                'age': get_col(f'AGE{year_str}X', 0),
                'sex': get_col('SEX', 0),
                'race': get_col('RACETHX', 0),
                'ed_visits': get_col(f'ERTOT{year_str}', 0).clip(0, 50),
                'insurance': get_col(f'INSCOV{year_str}', 0),
                'health_status': get_col(f'RTHLTH{year_str}', 3),
                'poverty_category': get_col(f'POVCAT{year_str}', 3),
                'education': get_col('EDUCYR', 12),
                'employment': get_col(f'EMPST{year_str}31', 0),
            })
            
            # Extract region
            region_col = f'REGION{year_str}'
            if region_col in df.columns:
                region = df[region_col]
                if region.dtype == 'object':
                    region = region.str.split().str[0]
                data['region'] = pd.to_numeric(region, errors='coerce').fillna(0)
            else:
                data['region'] = 0
            
            # Count chronic conditions
            chronic_count = 0
            for cond in ['CHDDX', 'DIABDX', 'HIBPDX', 'ASTHDX', 'ARTHDX', 'STRKDX']:
                if cond in df.columns:
                    chronic_count += (df[cond] == 1).astype(int)
            data['chronic_conditions'] = chronic_count
            
            years_data.append(data)
        except Exception as e:
            print(f"Error processing {year}: {e}")
            traceback.print_exc()
            continue
    
    # Combine all years
    all_data = pd.concat(years_data, ignore_index=True)
    
    # Clean
    all_data = all_data[all_data['age'].between(0, 120)]
    all_data = all_data[all_data['region'].isin([1, 2, 3, 4])]
    
    # Sort by person and year
    all_data = all_data.sort_values(['person_id', 'year'])
    
    # Create temporal features
    all_data['ed_visits_lag1'] = all_data.groupby('person_id')['ed_visits'].shift(1).fillna(0)
    all_data['ed_visits_lag2'] = all_data.groupby('person_id')['ed_visits'].shift(2).fillna(0)
    all_data['age_lag1'] = all_data.groupby('person_id')['age'].shift(1).fillna(0)
    all_data['age_lag2'] = all_data.groupby('person_id')['age'].shift(2).fillna(0)
    
    # Person-level aggregates
    all_data['person_ed_mean'] = all_data.groupby('person_id')['ed_visits'].transform('mean')
    all_data['person_ed_std'] = all_data.groupby('person_id')['ed_visits'].transform('std').fillna(0)
    all_data['person_observations'] = all_data.groupby('person_id')['person_id'].transform('count')
    
    # Sequence position
    all_data['sequence_position'] = all_data.groupby('person_id').cumcount()
    
    # Frequent ED user
    all_data['frequent_ed'] = (all_data['ed_visits'] >= 4).astype(int)
    
    print(f"\nMEPS data loaded: {len(all_data):,} person-years")
    print(f"  Unique persons: {all_data['person_id'].nunique():,}")
    print(f"  Years: {sorted(all_data['year'].unique())}")
    print(f"  Mean ED visits: {all_data['ed_visits'].mean():.2f}")
    print(f"  Frequent ED users: {all_data['frequent_ed'].mean()*100:.2f}%")
    
    return all_data


def prepare_train_test_split(data: pd.DataFrame, test_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data for COVID experiment
    Train: Pre-COVID (2019)
    Test: COVID shock (2020)
    """
    train_data = data[data['year'] < test_year].copy()
    test_data = data[data['year'] == test_year].copy()
    
    print(f"\nTrain/Test Split:")
    print(f"  Train (pre-{test_year}): {len(train_data):,} person-years")
    print(f"  Test ({test_year}): {len(test_data):,} person-years")
    
    return train_data, test_data


def save_prepared_data(data: pd.DataFrame, output_file: str = "rssm_meps_prepared.csv"):
    """Save prepared data"""
    data.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Load data
    data = load_meps_for_rssm()
    
    # Prepare train/test split
    train_data, test_data = prepare_train_test_split(data, test_year=2020)
    
    # Save
    save_prepared_data(data, "healthcare_world_model/rssm_meps_prepared.csv")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nNext: Train RSSM on 2019, test on 2020 COVID shock")
