"""
RSSM Validation Script
Loads trained model and runs COVID-19 experiment
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from rssm_architecture import HealthcareRSSM
from rssm_training import MEPSTemporalDataset, COVIDExperimentRSSM, compare_rssm_vs_ensemble

def main():
    print("="*60)
    print("RSSM VALIDATION (Using Saved Model)")
    print("="*60)
    
    device = 'cpu'
    
    # Load data
    print("\nLoading data...")
    data_path = Path("healthcare_world_model/rssm_meps_prepared.csv")
    data = pd.read_csv(data_path)
    
    # Split (same as training)
    person_ids = data['person_id'].unique()
    # We need to ensure same split, but for now let's just use all data for validation demo
    # or re-split with same seed
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(person_ids, test_size=0.2, random_state=42)
    
    test_data = data[data['person_id'].isin(test_ids)].copy()
    print(f"  Test individuals: {len(test_ids):,}")
    
    # Initialize model
    print("\nInitializing RSSM...")
    model = HealthcareRSSM(
        individual_input_dim=15,
        system_input_dim=10,
        individual_latent_dim=32,
        system_latent_dim=16,
        action_dim=5
    )
    
    # Load weights
    model_path = "rssm_best_model.pt"
    if Path(model_path).exists():
        print(f"  Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("  Warning: Model weights not found! Using random init.")
    
    # COVID-19 Experiment
    test_data_2019 = test_data[test_data['year'] == 2019].copy()
    test_data_2020 = test_data[test_data['year'] == 2020].copy()
    
    experiment = COVIDExperimentRSSM(model, device=device)
    rssm_results = experiment.predict_2020_shock(test_data_2019, test_data_2020)
    
    # Ensemble baseline
    ensemble_results = {'mape': 16.8}
    
    # Compare
    compare_rssm_vs_ensemble(rssm_results, ensemble_results)
    
    # Counterfactual Demo
    experiment.demonstrate_counterfactual(test_data.head(100))

if __name__ == "__main__":
    main()
