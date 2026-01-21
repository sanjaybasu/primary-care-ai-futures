
import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_meps_data(num_persons=1000, years=[2019, 2020, 2021, 2022]):
    """
    Generate synthetic MEPS-like data for pipeline verification.
    """
    print("Generating SYNTHETIC MEPS data...")
    
    rows = []
    
    for pid in range(num_persons):
        # Base attributes
        age_base = np.random.randint(0, 85)
        sex = np.random.randint(1, 3)
        race = np.random.randint(1, 5)
        region = np.random.randint(1, 5)
        
        # Temporal loop
        for year in years:
            age = age_base + (year - years[0])
            if age > 120: continue
            
            # Outcome generation (correlated with age/random)
            ed_visits_base = np.random.poisson(0.5 + (age/100))
            is_frequent = np.random.random() < 0.05
            ed_visits = np.random.randint(4, 10) if is_frequent else ed_visits_base
            
            # Other features
            insurance = np.random.randint(1, 4)
            health_status = np.random.randint(1, 6)
            poverty = np.random.randint(1, 6)
            education = 12
            employment = np.random.randint(1, 4)
            chronic = np.random.poisson(1 + (age/50))
            
            rows.append({
                'person_id': pid,
                'year': year,
                'panel': 1,
                'age': age,
                'sex': sex,
                'race': race,
                'ed_visits': ed_visits,
                'insurance': insurance,
                'health_status': health_status,
                'poverty_category': poverty,
                'education': education,
                'employment': employment,
                'region': region,
                'chronic_conditions': chronic
            })
            
    df = pd.DataFrame(rows)
    
    # Feature Engineering (mimic rssm_data_loader.py)
    df = df.sort_values(['person_id', 'year'])
    
    # Lags
    for lag in [1, 2]:
        df[f'ed_visits_lag{lag}'] = df.groupby('person_id')['ed_visits'].shift(lag).fillna(0)
        df[f'age_lag{lag}'] = df.groupby('person_id')['age'].shift(lag).fillna(0)
    
    # Aggregates
    df['person_ed_mean'] = df.groupby('person_id')['ed_visits'].transform('mean')
    df['person_ed_std'] = df.groupby('person_id')['ed_visits'].transform('std').fillna(0)
    df['person_observations'] = df.groupby('person_id')['person_id'].transform('count')
    df['sequence_position'] = df.groupby('person_id').cumcount()
    df['frequent_ed'] = (df['ed_visits'] >= 4).astype(int)
    
    # Add System Columns (filled with random realistic values for context)
    # rssm_training.py expects them or fills with 0
    df['population'] = 100000
    df['primary_care_physicians_per_100k'] = np.random.normal(70, 10, len(df))
    df['pct_broadband'] = np.random.uniform(0.5, 0.9, len(df)) # For simulation
    
    print(f"Generated {len(df)} rows.")
    return df

if __name__ == "__main__":
    df = generate_synthetic_meps_data()
    out_dir = Path("healthcare_world_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rssm_meps_prepared.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
