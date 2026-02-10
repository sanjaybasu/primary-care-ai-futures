
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

processed_dir = '/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/processed'
results_dir = '/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/results'

print("="*60)
print("MEDICAID EXPANSION: PREDICTIVE VALIDATION")
print("="*60)
print("Testing prediction of late-adopter outcomes (OK, MO) using 2019 baseline")

# Load state mortality data
df = pd.read_csv(f'{processed_dir}/state_mortality_2019_2022.csv')
exp_dates = pd.read_csv(f'{processed_dir}/medicaid_expansion_dates.csv')

# Merge expansion dates
df = pd.merge(df, exp_dates, on='state', how='left')

# Identify Late Adopters (Expanded in 2020 or 2021)
# Oklahoma (OK): July 2021
# Missouri (MO): Oct 2021
late_adopters = ['OK', 'MO']
controls = df[(df['expanded_medicaid'] == False) & (~df['state'].isin(late_adopters))]['state'].tolist()

print(f"Late Adopters (Test Set): {late_adopters}")
print(f"Controls (Non-expansion): {len(controls)} states")

# 1. World Model Prediction Logic
# We predict 2022 mortality for OK/MO assuming:
# - Baseline Dynamics (from 2019)
# - Intervention Effect (from systematic review/DiD priors, approx -4% to -10 deaths/100k)
# - COVID Shock (common shock)

# Calculate "Common Shock" from controls
control_df = df[df['state'].isin(controls)]
common_shock = (control_df['death_rate_2022'] - control_df['death_rate_2019']).mean()

print(f"Common Mortality Shock (Controls): +{common_shock:.1f} deaths/100k")

# Predict Late Adopters
validation_results = []

for state in late_adopters:
    baseline = df[df['state'] == state]['death_rate_2019'].values[0]
    actual_2022 = df[df['state'] == state]['death_rate_2022'].values[0]
    
    # Prediction A: Counterfactual (No Expansion)
    # 2019 + Common Shock
    pred_no_policy = baseline + common_shock
    
    # Prediction B: With Policy (World Model)
    # The world model applies the causal effect (-10.9 deaths/100k)
    policy_effect = -10.9
    pred_with_policy = baseline + common_shock + policy_effect
    
    # Calculate Errors
    error_no_policy = actual_2022 - pred_no_policy
    error_with_policy = actual_2022 - pred_with_policy
    
    print(f"\nState: {state}")
    print(f"  Baseline (2019): {baseline:.1f}")
    print(f"  Actual (2022):   {actual_2022:.1f}")
    print(f"  Pred (No Pol):   {pred_no_policy:.1f} (Error: {error_no_policy:.1f})")
    print(f"  Pred (Policy):   {pred_with_policy:.1f} (Error: {error_with_policy:.1f})")
    
    validation_results.append({
        'state': state,
        'baseline': baseline,
        'actual': actual_2022,
        'predicted_policy': pred_with_policy,
        'predicted_counterfactual': pred_no_policy,
        'error_policy': abs(error_with_policy),
        'error_counterfactual': abs(error_no_policy)
    })

# Summary
val_df = pd.DataFrame(validation_results)
mape_policy = (val_df['error_policy'] / val_df['actual']).mean() * 100
mape_counterfactual = (val_df['error_counterfactual'] / val_df['actual']).mean() * 100

print("\n" + "-"*40)
print("VALIDATION SUMMARY")
print("-"*(40))
print(f"MAPE (World Model / Policy): {mape_policy:.2f}%")
print(f"MAPE (Counterfactual / Null): {mape_counterfactual:.2f}%")

if mape_policy < mape_counterfactual:
    print("✓ Model with policy effect predicts outcomes better than null model.")
    improvement = (mape_counterfactual - mape_policy) / mape_counterfactual * 100
    print(f"  Predictive Improvement: {improvement:.1f}%")
else:
    print("! Null model predicts better (Policy effect may be overestimated or lagged).")

val_df.to_csv(f'{results_dir}/medicaid_predictive_validation_results.csv', index=False)
print(f"\nSaved results to {results_dir}/medicaid_predictive_validation_results.csv")
