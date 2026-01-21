"""
RSSM Medicaid Expansion Validation
Tests ability to capture heterogeneous treatment effects
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("Starting Medicaid Validation script...", flush=True)

from torch.utils.data import DataLoader, Dataset
from rssm_architecture import HealthcareRSSM
from rssm_training import RSSMTrainer
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error

class MedicaidDataset(Dataset):
    def __init__(self, data, sequence_length=2):
        self.data = data
        self.sequence_length = sequence_length
        self.person_ids = data['person_id'].unique()
        
        # Pre-group
        self.grouped = data.groupby('person_id')
        
    def __len__(self):
        return len(self.person_ids)
    
    def __getitem__(self, idx):
        person_id = self.person_ids[idx]
        group = self.grouped.get_group(person_id).sort_values('year')
        
        # Pad if needed (though we expect 2014-2015 pairs)
        if len(group) < self.sequence_length:
            # Pad with last observation
            pad = pd.concat([group.iloc[[-1]]] * (self.sequence_length - len(group)))
            group = pd.concat([group, pad], ignore_index=True)
        
        # Individual features: Age, ED Visits, Medicaid
        ind_features = torch.FloatTensor(group[['age', 'ed_visits', 'medicaid_any']].values)
        
        # System features: Expansion Group (One-hot or ordinal)
        # High=2, Mixed=1, Low=0
        exp_map = {'High': 2, 'Mixed': 1, 'Low': 0}
        exp_val = exp_map.get(group['expansion_group'].iloc[0], 0)
        
        # Create a simple system vector (Expansion Status, Region)
        # We'll just use expansion status for this test
        sys_features = torch.zeros(len(group), 10) # Match architecture dim
        sys_features[:, 0] = exp_val
        sys_features[:, 1] = group['region'].iloc[0]
        
        # Targets
        # We want to predict 2015 (next) from 2014 (current)
        # But we also want to reconstruct 2014
        
        # Current (2014)
        ed_current = group['ed_visits'].values[:-1]
        ed_next = group['ed_visits'].values[1:]
        
        # If sequence length is 2 (2014, 2015), we have 1 pair
        # But we pad to sequence length.
        # Let's just take the last step for simplicity in this validation
        
        targets = {
            'ed_visits': torch.FloatTensor([group['ed_visits'].iloc[0]]), # 2014
            'ed_visits_next': torch.FloatTensor([group['ed_visits'].iloc[-1]]), # 2015
            'frequent_ed': torch.FloatTensor([float(group['ed_visits'].iloc[0] >= 4)]),
            'medicaid': torch.FloatTensor([group['medicaid_any'].iloc[-1]]) # 2015
        }
        
        # Action (Empty for now)
        action = torch.zeros(5)
        
        return {
            'individual_obs': ind_features,
            'system_obs': sys_features,
            'action': action,
            'targets': targets
        }

def train_and_evaluate():
    print("="*60)
    print("MEDICAID EXPANSION VALIDATION")
    print("="*60)
    
    # Load data
    data_path = Path("healthcare_world_model/rssm_medicaid_prepared.csv")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows")
    
    # Filter for people present in both years for cleaner validation
    counts = data['person_id'].value_counts()
    valid_ids = counts[counts == 2].index
    data = data[data['person_id'].isin(valid_ids)].copy()
    print(f"Valid 2-year trajectories: {len(valid_ids):,}")
    
    # Split Train/Test
    # Train on 80% of people, Test on 20%
    # Stratified by Expansion Group
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(valid_ids, test_size=0.2, random_state=42)
    
    train_data = data[data['person_id'].isin(train_ids)]
    test_data = data[data['person_id'].isin(test_ids)]
    
    # Create Datasets
    train_ds = MedicaidDataset(train_data)
    test_ds = MedicaidDataset(test_data)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 1. Train Full RSSM (System-Aware)
    print("\nTraining System-Aware RSSM...")
    model = HealthcareRSSM(
        individual_input_dim=3,  # Age, ED, Medicaid
        system_input_dim=10,
        individual_latent_dim=16,
        system_latent_dim=8,
        action_dim=5
    )
    
    trainer = RSSMTrainer(model, device='cpu')
    model = trainer.train(train_loader, test_loader, epochs=5)
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    predictions = []
    actuals = []
    groups = []
    
    with torch.no_grad():
        for batch in test_loader:
            ind = batch['individual_obs']
            sys = batch['system_obs']
            target = batch['targets']
            
            # Forward pass
            # We want to predict t=1 (2015) given t=0 (2014)
            
            # Posterior for 2014
            post = model.encode(ind, sys)
            
            # Predict 2015
            # We need to transition from 2014 state
            # State at t=0 is just the posterior output (which is already the encoded state)
            state_0 = post
            
            # Transition
            prior_1 = model.transition_step(state_0['z_individual'], state_0['z_system'], torch.zeros(len(ind), 5))
            
            # Decode
            # We need to decode from prior_1 (which is the prediction for 2015)
            
            # Decode ED visits
            z_ind = prior_1['z_next_individual']
            ed_pred, _ = model.individual_decoder(z_ind)
            
            predictions.extend(ed_pred.squeeze().numpy())
            actuals.extend(target['ed_visits_next'][:, 0].numpy()) # Target is 2015
            
            # Get expansion group from system features
            # sys[:, 0, 0] is expansion group
            groups.extend(sys[:, 0, 0].numpy())
            
    # Analysis
    results = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions,
        'group': groups
    })
    
    # Map group back
    group_map = {2: 'High', 1: 'Mixed', 0: 'Low'}
    results['group_name'] = results['group'].map(group_map)
    
    print("\nResults by Expansion Group (RMSE):")
    for g in ['High', 'Low', 'Mixed']:
        sub = results[results['group_name'] == g]
        rmse = np.sqrt(mean_squared_error(sub['actual'], sub['predicted']))
        print(f"  {g}: {rmse:.4f}")
        
    # Compare to Baseline (Mean Predictor)
    print("\nBaseline (Mean) RMSE:")
    mean_pred = np.mean(train_data[train_data['year']==2015]['ed_visits'])
    for g in ['High', 'Low', 'Mixed']:
        sub = results[results['group_name'] == g]
        rmse = np.sqrt(mean_squared_error(sub['actual'], [mean_pred]*len(sub)))
        print(f"  {g}: {rmse:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
