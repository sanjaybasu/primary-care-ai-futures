"""
RSSM World Model - Data Preparation
Integrates MEPS (individual) + AHRF (system) + CHR (system) data
Creates unified temporal dataset for hierarchical RSSM training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import pyreadstat
from datetime import datetime

class RSSMDataPreparation:
    """Prepare integrated dataset for RSSM world model"""
    
    def __init__(self, data_dir: str = "healthcare_world_model/data"):
        self.data_dir = Path(data_dir)
        self.meps_dir = self.data_dir / "real_meps"
        self.hrsa_dir = Path("data/hrsa")  # HRSA is in root data dir
        self.chr_dir = Path("data/county_health_rankings")
        
        # Census region mapping
        self.region_map = {
            1: "Northeast",
            2: "Midwest", 
            3: "South",
            4: "West"
        }
        
    def load_meps_data(self) -> pd.DataFrame:
        """Load and process MEPS data (2019-2022)"""
        print("Loading MEPS data...")
        
        meps_files = {
            2019: "h216.dta",
            2020: "h224.dta", 
            2021: "h233.dta",
            2022: "h243.dta"
        }
        
        all_data = []
        
        for year, filename in meps_files.items():
            filepath = self.meps_dir / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found, skipping")
                continue
                
            print(f"  Loading {year}...")
            df, meta = pyreadstat.read_dta(str(filepath))
            
            # Extract key variables
            df_clean = pd.DataFrame({
                'person_id': df['DUPERSID'],
                'year': year,
                'panel': df['PANEL'],
                'age': df[f'AGE{str(year)[-2:]}X'],
                'sex': df['SEX'],
                'race': df['RACETHX'],
                'region': self._extract_region(df, year),
                'insurance': self._extract_insurance(df, year),
                'ed_visits': df[f'ERTOT{str(year)[-2:]}'],
                'health_status': self._extract_health_status(df, year),
                'income_category': self._extract_income(df, year),
                'chronic_conditions': self._count_chronic_conditions(df, year),
                'poverty_category': df.get(f'POVCAT{str(year)[-2:]}', 0),
                'education': df.get('EDUCYR', 0),
                'employment': df.get(f'EMPST{str(year)[-2:]}31', 0),
            })
            
            all_data.append(df_clean)
        
        meps_df = pd.concat(all_data, ignore_index=True)
        
        # Clean and validate
        meps_df = self._clean_meps_data(meps_df)
        
        print(f"MEPS data loaded: {len(meps_df):,} person-years")
        print(f"  Unique persons: {meps_df['person_id'].nunique():,}")
        print(f"  Years: {sorted(meps_df['year'].unique())}")
        
        return meps_df
    
    def _extract_region(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Extract region from MEPS data"""
        region_col = f'REGION{str(year)[-2:]}'
        if region_col not in df.columns:
            return pd.Series([np.nan] * len(df))
        
        region = df[region_col]
        
        # Handle categorical format "1 NORTHEAST" -> 1
        if region.dtype == 'object':
            region = region.str.split().str[0].astype(float)
        
        return region
    
    def _extract_insurance(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Extract insurance status"""
        inscov_col = f'INSCOV{str(year)[-2:]}'
        if inscov_col not in df.columns:
            return pd.Series([0] * len(df))
        
        # 1=Any private, 2=Public only, 3=Uninsured
        return df[inscov_col]
    
    def _extract_health_status(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Extract self-reported health status"""
        health_col = f'RTHLTH{str(year)[-2:]}'
        if health_col not in df.columns:
            return pd.Series([3] * len(df))  # Default to "Good"
        
        # 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor
        return df[health_col]
    
    def _extract_income(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Extract income category"""
        povcat_col = f'POVCAT{str(year)[-2:]}'
        if povcat_col not in df.columns:
            return pd.Series([3] * len(df))
        
        # 1=<100% FPL, 2=100-200%, 3=200-400%, 4=>400%
        return df[povcat_col]
    
    def _count_chronic_conditions(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Count chronic conditions"""
        # Common chronic condition indicators in MEPS
        conditions = [
            f'CHDDX',  # Heart disease
            f'DIABDX',  # Diabetes
            f'HIBPDX',  # Hypertension
            f'ASTHDX',  # Asthma
            f'ARTHDX',  # Arthritis
            f'STRKDX',  # Stroke
            f'CANCERDX',  # Cancer
        ]
        
        count = pd.Series([0] * len(df))
        for cond in conditions:
            if cond in df.columns:
                # 1=Yes, 2=No
                count += (df[cond] == 1).astype(int)
        
        return count
    
    def _clean_meps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean MEPS data"""
        # Remove negative values (MEPS uses negative for missing)
        numeric_cols = ['age', 'ed_visits', 'chronic_conditions']
        for col in numeric_cols:
            df[col] = df[col].clip(lower=0)
        
        # Remove invalid ages
        df = df[df['age'].between(0, 120)]
        
        # Remove invalid regions
        df = df[df['region'].isin([1, 2, 3, 4])]
        
        # Cap ED visits at reasonable maximum
        df['ed_visits'] = df['ed_visits'].clip(upper=50)
        
        return df
    
    def load_ahrf_data(self) -> pd.DataFrame:
        """Load and process AHRF county-level data"""
        print("Loading AHRF data...")
        
        ahrf_file = self.hrsa_dir / "AHRF_2023-2024.sas7bdat"
        if not ahrf_file.exists():
            print(f"Warning: AHRF file not found at {ahrf_file}")
            return pd.DataFrame()
        
        df, meta = pyreadstat.read_sas7bdat(str(ahrf_file))
        
        # Extract key variables for 2019-2022
        ahrf_data = []
        
        for year in [2019, 2020, 2021, 2022]:
            year_data = pd.DataFrame({
                'county_fips': df['F00002'],  # County FIPS
                'year': year,
                'population': df.get(f'F11984', 0),  # Total population
                'primary_care_physicians_per_100k': df.get(f'F12424', 0),
                'hospital_beds_per_1000': df.get(f'F08921', 0),
                'fqhc_count': df.get(f'F14842', 0),
                'rural_urban_code': df.get(f'F00011', 0),
                'metro_status': df.get(f'F00012', 0),
            })
            ahrf_data.append(year_data)
        
        ahrf_df = pd.concat(ahrf_data, ignore_index=True)
        
        # Clean
        ahrf_df = ahrf_df[ahrf_df['county_fips'] > 0]
        ahrf_df['county_fips'] = ahrf_df['county_fips'].astype(int)
        
        print(f"AHRF data loaded: {len(ahrf_df):,} county-years")
        print(f"  Unique counties: {ahrf_df['county_fips'].nunique():,}")
        
        return ahrf_df
    
    def load_chr_data(self) -> pd.DataFrame:
        """Load and process County Health Rankings data"""
        print("Loading County Health Rankings data...")
        
        chr_files = {
            2019: "2019_analytic_data.csv",
            2020: "2020_analytic_data.csv",
            2021: "2021_analytic_data.csv",
            2022: "2022_analytic_data.csv"
        }
        
        all_data = []
        
        for year, filename in chr_files.items():
            filepath = self.chr_dir / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found, skipping")
                continue
            
            print(f"  Loading {year}...")
            df = pd.read_csv(filepath)
            
            # Extract key variables
            df_clean = pd.DataFrame({
                'county_fips': df['fipscode'],
                'year': year,
                'premature_death': df.get('v001_rawvalue', np.nan),
                'poor_fair_health_pct': df.get('v002_rawvalue', np.nan),
                'adult_smoking_pct': df.get('v009_rawvalue', np.nan),
                'adult_obesity_pct': df.get('v011_rawvalue', np.nan),
                'uninsured_pct': df.get('v003_rawvalue', np.nan),
                'children_poverty_pct': df.get('v024_rawvalue', np.nan),
                'income_inequality': df.get('v044_rawvalue', np.nan),
                'preventable_hosp_stays': df.get('v005_rawvalue', np.nan),
            })
            
            all_data.append(df_clean)
        
        chr_df = pd.concat(all_data, ignore_index=True)
        
        # Clean
        chr_df = chr_df[chr_df['county_fips'] > 0]
        chr_df['county_fips'] = chr_df['county_fips'].astype(int)
        
        print(f"CHR data loaded: {len(chr_df):,} county-years")
        print(f"  Unique counties: {chr_df['county_fips'].nunique():,}")
        
        return chr_df
    
    def create_unified_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create unified dataset for RSSM training"""
        print("\n" + "="*60)
        print("Creating Unified RSSM Dataset")
        print("="*60 + "\n")
        
        # Load all data sources
        meps_df = self.load_meps_data()
        ahrf_df = self.load_ahrf_data()
        chr_df = self.load_chr_data()
        
        # Merge system-level data (AHRF + CHR)
        print("\nMerging system-level data...")
        system_df = ahrf_df.merge(
            chr_df,
            on=['county_fips', 'year'],
            how='outer'
        )
        
        print(f"System-level data: {len(system_df):,} county-years")
        
        # Create region-level aggregates for MEPS linkage
        print("\nCreating region-level aggregates...")
        
        # Map counties to Census regions (simplified - would need actual mapping)
        # For now, use state FIPS to infer region
        system_df['region'] = self._map_county_to_region(system_df['county_fips'])
        
        # Aggregate to region-year level
        region_df = system_df.groupby(['region', 'year']).agg({
            'population': 'sum',
            'primary_care_physicians_per_100k': 'mean',
            'hospital_beds_per_1000': 'mean',
            'fqhc_count': 'sum',
            'poor_fair_health_pct': 'mean',
            'adult_smoking_pct': 'mean',
            'adult_obesity_pct': 'mean',
            'uninsured_pct': 'mean',
            'children_poverty_pct': 'mean',
            'income_inequality': 'mean',
        }).reset_index()
        
        # Merge individual data with regional aggregates
        print("\nMerging individual and system data...")
        individual_df = meps_df.merge(
            region_df,
            on=['region', 'year'],
            how='left'
        )
        
        print(f"\nFinal individual-level dataset: {len(individual_df):,} person-years")
        print(f"Final system-level dataset: {len(system_df):,} county-years")
        
        # Create temporal sequences
        individual_df = self._create_temporal_sequences(individual_df)
        system_df = self._create_temporal_sequences_system(system_df)
        
        return individual_df, system_df
    
    def _map_county_to_region(self, county_fips: pd.Series) -> pd.Series:
        """Map county FIPS to Census region"""
        # State FIPS is first 2 digits
        state_fips = (county_fips / 1000).astype(int)
        
        # Simplified mapping (would need complete state->region mapping)
        region_mapping = {
            # Northeast: 1
            **{s: 1 for s in [9, 23, 25, 33, 44, 50, 34, 36, 42]},
            # Midwest: 2
            **{s: 2 for s in [17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46]},
            # South: 3
            **{s: 3 for s in [10, 11, 12, 13, 24, 37, 45, 51, 54, 1, 21, 28, 47, 5, 22, 40, 48]},
            # West: 4
            **{s: 4 for s in [4, 8, 16, 30, 32, 35, 49, 56, 2, 6, 15, 41, 53]},
        }
        
        return state_fips.map(region_mapping)
    
    def _create_temporal_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal sequences for RSSM training"""
        # Sort by person and year
        df = df.sort_values(['person_id', 'year'])
        
        # Create sequence indicators
        df['sequence_position'] = df.groupby('person_id').cumcount()
        df['sequence_length'] = df.groupby('person_id')['person_id'].transform('count')
        
        # Create lagged features
        for lag in [1, 2]:
            df[f'ed_visits_lag{lag}'] = df.groupby('person_id')['ed_visits'].shift(lag)
            df[f'age_lag{lag}'] = df.groupby('person_id')['age'].shift(lag)
        
        return df
    
    def _create_temporal_sequences_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal sequences for system-level data"""
        # Sort by county and year
        df = df.sort_values(['county_fips', 'year'])
        
        # Create sequence indicators
        df['sequence_position'] = df.groupby('county_fips').cumcount()
        df['sequence_length'] = df.groupby('county_fips')['county_fips'].transform('count')
        
        # Create lagged features
        for lag in [1]:
            df[f'population_lag{lag}'] = df.groupby('county_fips')['population'].shift(lag)
            df[f'uninsured_pct_lag{lag}'] = df.groupby('county_fips')['uninsured_pct'].shift(lag)
        
        return df
    
    def save_datasets(self, individual_df: pd.DataFrame, system_df: pd.DataFrame, 
                     output_dir: str = "healthcare_world_model"):
        """Save prepared datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        individual_file = output_path / "rssm_individual_data.csv"
        system_file = output_path / "rssm_system_data.csv"
        
        individual_df.to_csv(individual_file, index=False)
        system_df.to_csv(system_file, index=False)
        
        print(f"\nDatasets saved:")
        print(f"  Individual: {individual_file} ({len(individual_df):,} rows)")
        print(f"  System: {system_file} ({len(system_df):,} rows)")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Dataset Summary")
        print("="*60)
        
        print("\nIndividual-Level:")
        print(f"  Person-years: {len(individual_df):,}")
        print(f"  Unique persons: {individual_df['person_id'].nunique():,}")
        print(f"  Years: {sorted(individual_df['year'].unique())}")
        print(f"  Mean ED visits: {individual_df['ed_visits'].mean():.2f}")
        print(f"  Frequent ED users (≥4): {(individual_df['ed_visits'] >= 4).sum():,} ({(individual_df['ed_visits'] >= 4).mean()*100:.1f}%)")
        
        print("\nSystem-Level:")
        print(f"  County-years: {len(system_df):,}")
        print(f"  Unique counties: {system_df['county_fips'].nunique():,}")
        print(f"  Years: {sorted(system_df['year'].unique())}")
        print(f"  Mean uninsured %: {system_df['uninsured_pct'].mean():.1f}%")
        
        return individual_file, system_file


def main():
    """Main execution"""
    print("RSSM World Model - Data Preparation")
    print("="*60)
    
    # Initialize
    prep = RSSMDataPreparation()
    
    # Create unified dataset
    individual_df, system_df = prep.create_unified_dataset()
    
    # Save
    prep.save_datasets(individual_df, system_df)
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
