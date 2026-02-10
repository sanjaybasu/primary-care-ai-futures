
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

processed_dir = '/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/processed'
results_dir = '/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/results'

print("="*60)
print("TELEMEDICINE SURGE VALIDATION")
print("="*60)

# Load data (using state data as proxy for national trends if individual not available)
df = pd.read_csv(f'{processed_dir}/state_mortality_2019_2022.csv')

# Telemedicine "shock": In 2020, utilization went from <1% to ~30-50% in primary care
# We test if the model, given this structural break in "technology access", predicts observed mortality
# Note: Real-world mortality INCREASED due to COVID. The validation question is:
# Does the model credit Telemedicine with *mitigating* the increase, or having no effect?
# Evidence suggests telemedicine preserved access but didn't drastically reduce mortality itself (null effect).

# We will simulate the "Telemedicine Shock"
# We assume "Telemedicine Capacity" was a latent variable that spiked.

print("Loading data...")
# Synthetic proxy for telemedicine capacity (0 pre-2020, 1 post-2020)
# In a real World Model, this is an input vector.
# Here we test the specific hypothesis:
# Hypothesis: High-telemedicine states vs Low-telemedicine states difference in outcomes

# Since we don't have direct state-level telemedicine utilization ready-made in CSV,
# we will use the "Broadband Access" variable from SVI as a proxy for Telemedicine Potential.
raw_dir = '/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2/data/raw'
svi = pd.read_csv(f'{raw_dir}/svi_2022.csv', usecols=['ST_ABBR', 'E_NOINT', 'E_TOTPOP']) # E_NOINT = No Internet
svi_state = svi.groupby('ST_ABBR').apply(lambda x: np.average(x['E_NOINT'], weights=x['E_TOTPOP'])).reset_index(name='no_internet')
svi_state.columns = ['state', 'no_internet_pct']

# Merge
df = pd.merge(df, svi_state, on='state')

# Calculate Mortality Shock (2022 vs 2019)
df['excess_mortality'] = df['death_rate_2022'] - df['death_rate_2019']

# Telemedicine Potential = Inverse of No Internet
df['telemedicine_potential'] = 100 - df['no_internet_pct']

print("\n[1] VALIDATION: Does Telemedicine Potential Predict Lower Excess Mortality?")
correlation = df['excess_mortality'].corr(df['telemedicine_potential'])
print(f"Correlation (Telemedicine Potential vs Excess Mortality): {correlation:.3f}")

# Regression with controls (Medicaid expansion status as control)
import statsmodels.api as sm
X = df[['telemedicine_potential', 'expanded_medicaid']]
X['expanded_medicaid'] = X['expanded_medicaid'].astype(int)
X = sm.add_constant(X)
y = df['excess_mortality']

model = sm.OLS(y, X).fit()
print("\nRegression Results:")
print(model.summary())

tele_coeff = model.params['telemedicine_potential']
tele_p = model.pvalues['telemedicine_potential']
print(f"\nTelemedicine Potential Effect: {tele_coeff:.2f} (p={tele_p:.3f})")

# Interpretation
print("\nINTERPRETATION:")
if tele_p > 0.05:
    print("✓ Validation Successful (Null Effect): Telemedicine potential did NOT significantly predict mortality variation.")
    print("  This aligns with our systematic review findings of null mortality effects for tech interventions.")
else:
    if tele_coeff < 0:
        print("! Signal Detected: Higher telemedicine potential associated with LOWER excess mortality.")
    else:
        print("! Signal Detected: Higher telemedicine potential associated with HIGHER excess mortality.")

# Save results
pd.DataFrame([{
    'validation_type': 'Telemedicine Surge',
    'metric': 'Coefficient (Telemedicine Potential)',
    'value': tele_coeff,
    'p_value': tele_p,
    'conclusion': 'Null effect' if tele_p > 0.05 else 'Significant effect'
}]).to_csv(f'{results_dir}/telemedicine_validation.csv', index=False)

print(f"\nSaved results to {results_dir}/telemedicine_validation.csv")
