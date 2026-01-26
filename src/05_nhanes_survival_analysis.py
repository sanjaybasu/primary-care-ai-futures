#!/usr/bin/env python3
"""
Individual-level survival analysis using NHANES with public mortality linkage.

This is REAL ANALYSIS using actual individual-level mortality data from NCHS.
NHANES Public-Use Linked Mortality Files provide mortality follow-up through 2019.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDIVIDUAL_DIR = os.path.join(DATA_DIR, 'individual')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("=" * 70)
print("NHANES INDIVIDUAL-LEVEL SURVIVAL ANALYSIS")
print("Using Public-Use Linked Mortality Files (NCHS)")
print("=" * 70)

# ============================================================================
# 1. LOAD NHANES MORTALITY DATA
# ============================================================================
print("\n[1/5] Loading NHANES mortality linkage data...")

mort_df = pd.read_csv(os.path.join(INDIVIDUAL_DIR, 'nhanes_mortality_linked.csv'),
                       dtype={'SEQN': str})

print(f"    Total records: {len(mort_df)}")
print(f"    Survey cycles: {mort_df['survey_cycle'].unique()}")

# Convert to numeric
for col in ['ELIGSTAT', 'MORTSTAT', 'DODYEAR']:
    mort_df[col] = pd.to_numeric(mort_df[col], errors='coerce')

# Filter to eligible for mortality follow-up (ELIGSTAT=1)
mort_df = mort_df[mort_df['ELIGSTAT'] == 1].copy()
print(f"    Eligible for follow-up: {len(mort_df)}")

# Create mortality indicator
mort_df['died'] = (mort_df['MORTSTAT'] == 1).astype(int)
print(f"    Deaths: {mort_df['died'].sum()}")
print(f"    Crude mortality: {mort_df['died'].mean()*100:.2f}%")

# ============================================================================
# 2. DOWNLOAD AND MERGE NHANES DEMOGRAPHIC DATA
# ============================================================================
print("\n[2/5] Downloading NHANES demographic data...")

import requests
from io import BytesIO

# NHANES demographic files (corrected URLs)
demo_urls = {
    '2017_2018': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT',
    '2015_2016': 'https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT',
    '2013_2014': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT',
    '2011_2012': 'https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT',
    '2009_2010': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT',
    '2007_2008': 'https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/DEMO_E.XPT',
    '2005_2006': 'https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/DEMO_D.XPT',
    '2003_2004': 'https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DEMO_C.XPT',
    '2001_2002': 'https://wwwn.cdc.gov/Nchs/Nhanes/2001-2002/DEMO_B.XPT',
    '1999_2000': 'https://wwwn.cdc.gov/Nchs/Nhanes/1999-2000/DEMO.XPT',
}

# Health insurance files
hiq_urls = {
    '2017_2018': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HIQ_J.XPT',
    '2015_2016': 'https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HIQ_I.XPT',
    '2013_2014': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/HIQ_H.XPT',
    '2011_2012': 'https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/HIQ_G.XPT',
    '2009_2010': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/HIQ_F.XPT',
    '2007_2008': 'https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/HIQ_E.XPT',
    '2005_2006': 'https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/HIQ_D.XPT',
    '2003_2004': 'https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/HIQ_C.XPT',
    '2001_2002': 'https://wwwn.cdc.gov/Nchs/Nhanes/2001-2002/HIQ_B.XPT',
    '1999_2000': 'https://wwwn.cdc.gov/Nchs/Nhanes/1999-2000/HIQ.XPT',
}

# Healthcare utilization files
huq_urls = {
    '2017_2018': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HUQ_J.XPT',
    '2015_2016': 'https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HUQ_I.XPT',
    '2013_2014': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/HUQ_H.XPT',
    '2011_2012': 'https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/HUQ_G.XPT',
    '2009_2010': 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/HUQ_F.XPT',
    '2007_2008': 'https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/HUQ_E.XPT',
    '2005_2006': 'https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/HUQ_D.XPT',
    '2003_2004': 'https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/HUQ_C.XPT',
    '2001_2002': 'https://wwwn.cdc.gov/Nchs/Nhanes/2001-2002/HUQ_B.XPT',
    '1999_2000': 'https://wwwn.cdc.gov/Nchs/Nhanes/1999-2000/HUQ.XPT',
}

def download_xpt(url):
    """Download and read XPT file."""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return pd.read_sas(BytesIO(response.content), format='xport')
    except Exception as e:
        pass
    return None

all_demo = []
all_hiq = []
all_huq = []

for cycle in demo_urls.keys():
    print(f"    Downloading {cycle}...", end=" ")

    demo = download_xpt(demo_urls[cycle])
    hiq = download_xpt(hiq_urls[cycle])
    huq = download_xpt(huq_urls[cycle])

    if demo is not None:
        demo['survey_cycle'] = cycle
        all_demo.append(demo)
        print(f"DEMO: {len(demo)}", end=" ")

    if hiq is not None:
        hiq['survey_cycle'] = cycle
        all_hiq.append(hiq)
        print(f"HIQ: {len(hiq)}", end=" ")

    if huq is not None:
        huq['survey_cycle'] = cycle
        all_huq.append(huq)
        print(f"HUQ: {len(huq)}", end=" ")

    print()

# Combine all cycles
demo_df = pd.concat(all_demo, ignore_index=True) if all_demo else None
hiq_df = pd.concat(all_hiq, ignore_index=True) if all_hiq else None
huq_df = pd.concat(all_huq, ignore_index=True) if all_huq else None

print(f"\n    Total demographic records: {len(demo_df) if demo_df is not None else 0}")
print(f"    Total insurance records: {len(hiq_df) if hiq_df is not None else 0}")
print(f"    Total utilization records: {len(huq_df) if huq_df is not None else 0}")

# ============================================================================
# 3. MERGE DATASETS
# ============================================================================
print("\n[3/5] Merging datasets...")

# Convert SEQN to string for merging
mort_df['SEQN'] = mort_df['SEQN'].astype(str).str.strip()

if demo_df is not None:
    demo_df['SEQN'] = demo_df['SEQN'].astype(str).str.strip()
    # Merge mortality with demographics
    analysis_df = mort_df.merge(demo_df, on=['SEQN', 'survey_cycle'], how='inner')
    print(f"    After demo merge: {len(analysis_df)}")
else:
    analysis_df = mort_df.copy()

if hiq_df is not None:
    hiq_df['SEQN'] = hiq_df['SEQN'].astype(str).str.strip()
    analysis_df = analysis_df.merge(hiq_df[['SEQN', 'survey_cycle', 'HIQ011']],
                                     on=['SEQN', 'survey_cycle'], how='left')

if huq_df is not None:
    huq_df['SEQN'] = huq_df['SEQN'].astype(str).str.strip()
    # HUQ050 = usual place for healthcare
    # HUQ010 = general health condition
    cols_to_merge = ['SEQN', 'survey_cycle']
    if 'HUQ050' in huq_df.columns:
        cols_to_merge.append('HUQ050')
    if 'HUQ010' in huq_df.columns:
        cols_to_merge.append('HUQ010')
    analysis_df = analysis_df.merge(huq_df[cols_to_merge],
                                     on=['SEQN', 'survey_cycle'], how='left')

print(f"    Final analysis sample: {len(analysis_df)}")

# ============================================================================
# 4. CREATE ANALYSIS VARIABLES
# ============================================================================
print("\n[4/5] Creating analysis variables...")

# Age (RIDAGEYR)
if 'RIDAGEYR' in analysis_df.columns:
    analysis_df['age'] = pd.to_numeric(analysis_df['RIDAGEYR'], errors='coerce')
    # Create age groups
    analysis_df['age_group'] = pd.cut(analysis_df['age'],
                                       bins=[0, 45, 65, 120],
                                       labels=['18-44', '45-64', '65+'])

# Sex (RIAGENDR: 1=Male, 2=Female)
if 'RIAGENDR' in analysis_df.columns:
    analysis_df['female'] = (analysis_df['RIAGENDR'] == 2).astype(int)

# Race/ethnicity (RIDRETH1 or RIDRETH3)
race_col = 'RIDRETH3' if 'RIDRETH3' in analysis_df.columns else 'RIDRETH1'
if race_col in analysis_df.columns:
    analysis_df['race_eth'] = analysis_df[race_col].map({
        1: 'Mexican American',
        2: 'Other Hispanic',
        3: 'Non-Hispanic White',
        4: 'Non-Hispanic Black',
        6: 'Non-Hispanic Asian',
        7: 'Other/Multi'
    })

# Education (DMDEDUC2 for adults)
if 'DMDEDUC2' in analysis_df.columns:
    analysis_df['college'] = (analysis_df['DMDEDUC2'] >= 4).astype(int)

# Insurance status (HIQ011: 1=Yes covered, 2=No)
if 'HIQ011' in analysis_df.columns:
    analysis_df['insured'] = (analysis_df['HIQ011'] == 1).astype(int)
    analysis_df['uninsured'] = (analysis_df['HIQ011'] == 2).astype(int)

# Usual place for care (HUQ050: 1=Yes)
if 'HUQ050' in analysis_df.columns:
    analysis_df['usual_care'] = (analysis_df['HUQ050'] == 1).astype(int)

# Poverty income ratio (INDFMPIR)
if 'INDFMPIR' in analysis_df.columns:
    analysis_df['pir'] = pd.to_numeric(analysis_df['INDFMPIR'], errors='coerce')
    analysis_df['poor'] = (analysis_df['pir'] < 1).astype(int)
    analysis_df['low_income'] = (analysis_df['pir'] < 2).astype(int)

# Survey year for analysis
analysis_df['survey_year'] = analysis_df['survey_cycle'].str[:4].astype(int)

# Filter to adults
analysis_df = analysis_df[analysis_df['age'] >= 18].copy()
print(f"    Adults (18+): {len(analysis_df)}")
print(f"    Deaths: {analysis_df['died'].sum()}")

# ============================================================================
# 5. SURVIVAL ANALYSIS
# ============================================================================
print("\n[5/5] Survival analysis...")

print("\n" + "-" * 50)
print("MORTALITY BY INSURANCE STATUS")
print("-" * 50)

if 'insured' in analysis_df.columns:
    # Mortality by insurance status
    insured = analysis_df[analysis_df['insured'] == 1]
    uninsured = analysis_df[analysis_df['uninsured'] == 1]

    print(f"\n  Insured (n={len(insured)}):")
    print(f"    Deaths: {insured['died'].sum()}")
    print(f"    Mortality: {insured['died'].mean()*100:.2f}%")

    print(f"\n  Uninsured (n={len(uninsured)}):")
    print(f"    Deaths: {uninsured['died'].sum()}")
    print(f"    Mortality: {uninsured['died'].mean()*100:.2f}%")

    # Risk ratio
    rr_insurance = uninsured['died'].mean() / insured['died'].mean()
    print(f"\n  Risk Ratio (Uninsured/Insured): {rr_insurance:.3f}")

    # Chi-square test
    contingency = pd.crosstab(analysis_df['insured'], analysis_df['died'])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    print(f"  Chi-square p-value: {p_val:.6f}")

print("\n" + "-" * 50)
print("MORTALITY BY USUAL SOURCE OF CARE")
print("-" * 50)

if 'usual_care' in analysis_df.columns:
    has_usual = analysis_df[analysis_df['usual_care'] == 1]
    no_usual = analysis_df[analysis_df['usual_care'] == 0]

    print(f"\n  Has usual source (n={len(has_usual)}):")
    print(f"    Deaths: {has_usual['died'].sum()}")
    print(f"    Mortality: {has_usual['died'].mean()*100:.2f}%")

    print(f"\n  No usual source (n={len(no_usual)}):")
    print(f"    Deaths: {no_usual['died'].sum()}")
    print(f"    Mortality: {no_usual['died'].mean()*100:.2f}%")

    rr_usual = no_usual['died'].mean() / has_usual['died'].mean() if has_usual['died'].mean() > 0 else np.nan
    print(f"\n  Risk Ratio (No usual/Has usual): {rr_usual:.3f}")

print("\n" + "-" * 50)
print("MORTALITY BY POVERTY STATUS")
print("-" * 50)

if 'poor' in analysis_df.columns:
    poor = analysis_df[analysis_df['poor'] == 1]
    not_poor = analysis_df[analysis_df['poor'] == 0]

    print(f"\n  In poverty (n={len(poor)}):")
    print(f"    Deaths: {poor['died'].sum()}")
    print(f"    Mortality: {poor['died'].mean()*100:.2f}%")

    print(f"\n  Not in poverty (n={len(not_poor)}):")
    print(f"    Deaths: {not_poor['died'].sum()}")
    print(f"    Mortality: {not_poor['died'].mean()*100:.2f}%")

    rr_poverty = poor['died'].mean() / not_poor['died'].mean() if not_poor['died'].mean() > 0 else np.nan
    print(f"\n  Risk Ratio (Poor/Not poor): {rr_poverty:.3f}")

print("\n" + "-" * 50)
print("MORTALITY BY AGE GROUP")
print("-" * 50)

if 'age_group' in analysis_df.columns:
    for ag in ['18-44', '45-64', '65+']:
        subset = analysis_df[analysis_df['age_group'] == ag]
        if len(subset) > 0:
            print(f"\n  Age {ag} (n={len(subset)}):")
            print(f"    Deaths: {subset['died'].sum()}")
            print(f"    Mortality: {subset['died'].mean()*100:.2f}%")

# ============================================================================
# LOGISTIC REGRESSION FOR ADJUSTED ODDS RATIOS
# ============================================================================
print("\n" + "-" * 50)
print("MULTIVARIABLE LOGISTIC REGRESSION")
print("-" * 50)

try:
    from scipy.special import expit, logit
    import statsmodels.api as sm

    # Prepare regression data
    reg_vars = ['died', 'insured', 'usual_care', 'age', 'female', 'poor']
    reg_df = analysis_df[reg_vars].dropna()

    print(f"\n  Regression sample: {len(reg_df)}")

    # Fit logistic regression
    X = reg_df[['insured', 'usual_care', 'age', 'female', 'poor']]
    X = sm.add_constant(X)
    y = reg_df['died']

    model = sm.Logit(y, X).fit(disp=0)

    print("\n  Logistic Regression Results (Outcome: Death)")
    print("  " + "=" * 55)
    print(f"  {'Variable':<20} {'OR':>10} {'95% CI':>20} {'p-value':>10}")
    print("  " + "-" * 55)

    for var in ['insured', 'usual_care', 'age', 'female', 'poor']:
        coef = model.params[var]
        se = model.bse[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96*se)
        ci_high = np.exp(coef + 1.96*se)
        pval = model.pvalues[var]
        print(f"  {var:<20} {or_val:>10.3f} ({ci_low:>6.3f}, {ci_high:>6.3f}) {pval:>10.4f}")

    print("  " + "=" * 55)

    # Key findings
    print("\n  KEY FINDING - Adjusted Odds Ratios:")
    ins_or = np.exp(model.params['insured'])
    ins_ci_low = np.exp(model.params['insured'] - 1.96*model.bse['insured'])
    ins_ci_high = np.exp(model.params['insured'] + 1.96*model.bse['insured'])
    print(f"    Insurance: OR = {ins_or:.3f} (95% CI: {ins_ci_low:.3f}-{ins_ci_high:.3f})")
    print(f"    Interpretation: Insurance associated with {(1-ins_or)*100:.1f}% lower odds of death")

    if 'usual_care' in model.params:
        usc_or = np.exp(model.params['usual_care'])
        usc_ci_low = np.exp(model.params['usual_care'] - 1.96*model.bse['usual_care'])
        usc_ci_high = np.exp(model.params['usual_care'] + 1.96*model.bse['usual_care'])
        print(f"    Usual source of care: OR = {usc_or:.3f} (95% CI: {usc_ci_low:.3f}-{usc_ci_high:.3f})")

except Exception as e:
    print(f"  Regression failed: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save analysis dataset
analysis_df.to_csv(os.path.join(RESULTS_DIR, 'nhanes_analysis_dataset.csv'), index=False)
print(f"\n  Saved analysis dataset: {len(analysis_df)} records")

# Summary statistics
summary = {
    'Metric': [
        'Total Sample',
        'Deaths',
        'Crude Mortality Rate',
        'Insured N',
        'Uninsured N',
        'Insured Mortality',
        'Uninsured Mortality',
        'Risk Ratio (Uninsured/Insured)',
        'Insurance Adjusted OR (95% CI)'
    ],
    'Value': [
        len(analysis_df),
        analysis_df['died'].sum(),
        f"{analysis_df['died'].mean()*100:.2f}%",
        len(analysis_df[analysis_df['insured']==1]) if 'insured' in analysis_df.columns else 'N/A',
        len(analysis_df[analysis_df['uninsured']==1]) if 'uninsured' in analysis_df.columns else 'N/A',
        f"{insured['died'].mean()*100:.2f}%" if 'insured' in analysis_df.columns else 'N/A',
        f"{uninsured['died'].mean()*100:.2f}%" if 'uninsured' in analysis_df.columns else 'N/A',
        f"{rr_insurance:.3f}" if 'rr_insurance' in dir() else 'N/A',
        f"{ins_or:.3f} ({ins_ci_low:.3f}-{ins_ci_high:.3f})" if 'ins_or' in dir() else 'N/A'
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_DIR, 'nhanes_survival_summary.csv'), index=False)

print(f"\n  Saved summary: {os.path.join(RESULTS_DIR, 'nhanes_survival_summary.csv')}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
