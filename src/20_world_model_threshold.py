#!/usr/bin/env python3
"""
World Model Threshold Analysis

Core Innovation: For each intervention, determine what effect magnitude would be 
required to close X% of the primary care access-mortality gap, then compare to 
observed effects from meta-analyses.

This script answers the key policy question:
"What interventions—and at what effect magnitudes—are required to close 
primary care access gaps and associated mortality disparities?"
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2")
META_DIR = BASE_DIR / "meta_analysis" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# EMPIRICAL FOUNDATIONS: Access-Mortality Relationship
# ==============================================================================

# From our causal inference analysis and published literature:
# Key empirical estimates linking access to mortality

EMPIRICAL_FOUNDATIONS = {
    # Medicaid expansion (increases access by ~10 percentage points)
    # Effect: -10.9 deaths per 100k (our DiD estimate)
    'coverage_increase_10pp': {
        'mortality_reduction': 10.9,  # deaths per 100k
        'mortality_reduction_ci': (0.7, 20.6),  # 95% CI from bootstrap
        'source': 'Our DiD analysis (2019-2022)',
        'mechanism': 'Insurance coverage → access → utilization → outcomes',
    },
    
    # PCP supply (from Bailey & Goodman-Bacon 2015, CHC analysis)
    # FQHCs associated with 2-7% mortality reduction for beneficiaries
    'pcp_per_100k_increase': {
        'mortality_reduction_per_10_pcp': 15.0,  # estimated from FQHC literature
        'mortality_reduction_ci': (5.0, 25.0),
        'source': 'Bailey & Goodman-Bacon 2015 AER',
        'mechanism': 'Supply → access → preventive care → reduced preventable mortality',
    },
    
    # From Basu et al. (2021) JAMA Internal Medicine
    # Primary care workforce shortage → 7,000-15,000 excess deaths nationally
    'workforce_shortage_mortality': {
        'excess_deaths_per_year': 11000,  # midpoint estimate
        'source': 'Basu et al. 2021 Ann Intern Med',
        'mechanism': 'Shortage → unmet need → delayed care → preventable mortality',
    },
}

# ==============================================================================
# INTERVENTION EFFECTS FROM META-ANALYSES
# ==============================================================================

# These come from our verified meta-analyses and systematic reviews
INTERVENTION_EFFECTS = {
    # HIGH GRADE evidence
    'Integrated Behavioral Health': {
        'metric': 'OR depression response',
        'effect': 2.52,
        'ci_low': 2.11,
        'ci_high': 3.01,
        'grade': 'HIGH',
        'access_pathway': 'Depression treatment → reduced psychiatric ED visits, improved engagement',
        'estimated_mortality_benefit': 'Indirect via QoL and suicide prevention',
        'scalability': 'High - can be implemented in existing practices',
    },
    
    'Clinical Pharmacist - Heart Failure': {
        'metric': 'RR mortality',
        'effect': 0.67,
        'ci_low': 0.57,
        'ci_high': 0.80,
        'grade': 'HIGH',
        'access_pathway': 'Medication optimization → reduced HF exacerbations → reduced mortality',
        'estimated_mortality_benefit': '33% reduction in HF mortality',
        'scalability': 'Moderate - requires pharmacist integration',
    },
    
    # MODERATE GRADE evidence
    'Clinical Pharmacist - Hypertension': {
        'metric': 'SBP reduction (mmHg)',
        'effect': -7.6,
        'ci_low': -8.6,
        'ci_high': -6.5,
        'grade': 'MODERATE',
        'access_pathway': 'BP control → reduced stroke, MI, kidney disease',
        'estimated_mortality_benefit': '~10% CVD mortality reduction per 10 mmHg SBP',
        'scalability': 'Moderate',
    },
    
    'Community Health Workers - Diabetes': {
        'metric': 'HbA1c reduction (%)',
        'effect': -0.26,
        'ci_low': -0.36,
        'ci_high': -0.15,
        'grade': 'MODERATE',
        'access_pathway': 'Engagement → adherence → glycemic control → reduced complications',
        'estimated_mortality_benefit': '5-10% microvascular complication reduction',
        'scalability': 'High - low-cost workforce expansion',
    },
    
    'School-Based Health Centers - Mental Health': {
        'metric': 'OR mental health access',
        'effect': 1.91,
        'ci_low': 1.65,
        'ci_high': 2.22,
        'grade': 'MODERATE',
        'access_pathway': 'Youth access → early intervention → prevented chronic conditions',
        'estimated_mortality_benefit': 'Long-term via life-course effects',
        'scalability': 'High in school settings',
    },
    
    # LOW GRADE evidence (AI interventions)
    'AI Ambient Scribes': {
        'metric': 'OR burnout',
        'effect': 0.26,
        'ci_low': 0.13,
        'ci_high': 0.54,
        'grade': 'LOW',
        'access_pathway': 'Reduced burnout → retention → maintained access',
        'estimated_mortality_benefit': 'Indirect via workforce retention',
        'scalability': 'High - technology-based',
    },
    
    'AI Symptom Checkers': {
        'metric': 'Triage accuracy (%)',
        'effect': 57,  # appropriate triage
        'ci_low': 52,
        'ci_high': 61,
        'grade': 'LOW',
        'access_pathway': 'Self-triage → appropriate care-seeking → efficient utilization',
        'estimated_mortality_benefit': 'Uncertain - safety concerns remain',
        'scalability': 'High - consumer-facing technology',
    },
    
    # Traditional policy interventions (from literature)
    'Medicaid Expansion': {
        'metric': 'Deaths per 100k reduction',
        'effect': 10.9,
        'ci_low': 0.7,
        'ci_high': 20.6,
        'grade': 'MODERATE',
        'access_pathway': 'Coverage → access → utilization → outcomes',
        'estimated_mortality_benefit': 'Direct mortality reduction',
        'scalability': 'State policy decision',
    },
}

# ==============================================================================
# THRESHOLD ANALYSIS
# ==============================================================================

def calculate_threshold_analysis():
    """
    Calculate what effect sizes are needed to close various proportions 
    of the primary care access-mortality gap.
    """
    
    # Define the "gap"
    # National mortality disparity between high-access and low-access areas
    # Based on SVI analysis: Q4 (high vulnerability) vs Q1 (low vulnerability)
    # Typical gap: ~150-200 deaths per 100k difference
    MORTALITY_GAP = 175  # deaths per 100k between most and least vulnerable quintiles
    
    # Target gap closures
    target_closures = [0.15, 0.25, 0.50]  # 15%, 25%, 50%
    
    results = []
    
    print("=" * 80)
    print("WORLD MODEL THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"\nBaseline mortality gap (Q4 vs Q1 vulnerability): {MORTALITY_GAP} deaths per 100k")
    
    for intervention_name, data in INTERVENTION_EFFECTS.items():
        print(f"\n{'-' * 60}")
        print(f"INTERVENTION: {intervention_name}")
        print(f"{'-' * 60}")
        print(f"  Observed effect: {data['effect']} {data['metric']}")
        print(f"  95% CI: ({data['ci_low']}, {data['ci_high']})")
        print(f"  GRADE: {data['grade']}")
        print(f"  Pathway: {data['access_pathway']}")
        
        # Translate effect to mortality impact
        # This requires assumptions about the causal pathway
        mortality_impact = translate_to_mortality(intervention_name, data)
        
        if mortality_impact is not None:
            print(f"  Estimated mortality impact: {mortality_impact:.1f} deaths per 100k")
            
            for target in target_closures:
                required_reduction = MORTALITY_GAP * target
                multiplier = required_reduction / mortality_impact if mortality_impact > 0 else np.inf
                sufficient = mortality_impact >= required_reduction
                
                results.append({
                    'intervention': intervention_name,
                    'observed_effect': data['effect'],
                    'effect_metric': data['metric'],
                    'grade': data['grade'],
                    'estimated_mortality_impact': mortality_impact,
                    'target_gap_closure': f"{int(target*100)}%",
                    'required_reduction': required_reduction,
                    'required_multiplier': multiplier,
                    'sufficient_alone': sufficient,
                })
                
                print(f"  To close {int(target*100)}% of gap ({required_reduction:.0f} deaths/100k):")
                if sufficient:
                    print(f"    ✓ SUFFICIENT: Observed effect meets target")
                else:
                    print(f"    ✗ INSUFFICIENT: Would need {multiplier:.1f}x larger effect")
    
    return pd.DataFrame(results)

def translate_to_mortality(intervention_name, data):
    """
    Translate intervention effect to estimated mortality impact.
    
    This is the key modeling step that connects intervention effects
    (often on intermediate outcomes) to mortality.
    """
    
    # Direct mortality interventions
    if 'Medicaid Expansion' in intervention_name:
        return data['effect']  # Already in deaths per 100k
    
    if 'Heart Failure' in intervention_name:
        # RR 0.67 = 33% reduction in HF mortality
        # HF mortality is ~20 per 100k nationally
        hf_baseline_mortality = 20
        reduction = (1 - data['effect']) * hf_baseline_mortality
        return reduction  # ~6.6 per 100k
    
    if 'Hypertension' in intervention_name:
        # 10 mmHg SBP reduction → ~10-15% CVD mortality reduction
        # CVD mortality is ~170 per 100k
        cvd_mortality = 170
        sbp_reduction = abs(data['effect'])
        cvd_reduction_pct = sbp_reduction * 0.01  # ~1% per mmHg
        return cvd_mortality * cvd_reduction_pct  # ~13 per 100k
    
    if 'Diabetes' in intervention_name:
        # 0.26% HbA1c reduction → ~5% microvascular complication reduction
        # Diabetes mortality ~25 per 100k
        dm_mortality = 25
        complication_reduction = 0.05  # 5%
        return dm_mortality * complication_reduction  # ~1.25 per 100k
    
    if 'Behavioral Health' in intervention_name:
        # Depression treatment → reduced suicide, reduced cardiovascular
        # Suicide mortality ~14 per 100k, depression treatment may prevent ~10%
        suicide_mortality = 14
        depression_contribution = 0.3  # ~30% of suicides have depression
        treatment_effect = 0.3  # Assume 30% of at-risk population would benefit
        return suicide_mortality * depression_contribution * treatment_effect  # ~1.3 per 100k
    
    if 'AI Ambient Scribes' in intervention_name:
        # Burnout reduction → retention → maintained access
        # Very indirect - estimate via workforce effect
        # 74% burnout reduction, assume 20% of burned out would leave
        # Loss of 10% of workforce → ~10 excess deaths per 100k over time
        workforce_mortality_impact = 10
        burnout_contribution = 0.2  # 20% of potential attrition
        return workforce_mortality_impact * burnout_contribution  # ~2 per 100k
    
    if 'AI Symptom Checkers' in intervention_name:
        # Uncertain - could prevent delays OR cause false reassurance
        # Net effect unclear
        return None  # Insufficient evidence for mortality translation
    
    if 'School-Based' in intervention_name:
        # Long-term life-course effects - very delayed benefit
        # Minimal short-term mortality impact
        return 0.5  # Nominal small effect
    
    return None

def create_threshold_figure(results_df):
    """Create visualization of threshold analysis."""
    import matplotlib.pyplot as plt
    
    # Filter to interventions with mortality translation
    plotable = results_df[results_df['estimated_mortality_impact'].notna()].copy()
    
    # Get unique interventions and their impacts
    interventions = plotable.groupby('intervention').first().reset_index()
    interventions = interventions.sort_values('estimated_mortality_impact', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart
    y_pos = range(len(interventions))
    
    # Color by GRADE
    colors = {'HIGH': '#2ecc71', 'MODERATE': '#f1c40f', 'LOW': '#e74c3c'}
    bar_colors = [colors.get(g, 'gray') for g in interventions['grade']]
    
    bars = ax.barh(y_pos, interventions['estimated_mortality_impact'], color=bar_colors, alpha=0.8)
    
    # Reference lines for gap closure targets
    gap = 175
    ax.axvline(x=gap * 0.15, color='blue', linestyle='--', linewidth=2, label='15% gap closure')
    ax.axvline(x=gap * 0.25, color='orange', linestyle='--', linewidth=2, label='25% gap closure')
    ax.axvline(x=gap * 0.50, color='red', linestyle='--', linewidth=2, label='50% gap closure')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(interventions['intervention'])
    ax.set_xlabel('Estimated Mortality Impact (deaths per 100k)')
    ax.set_title('Threshold Analysis: Intervention Effects vs Required Gap Closure\n(Color indicates GRADE quality: Green=HIGH, Yellow=MODERATE, Red=LOW)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    fig_path = RESULTS_DIR / "threshold_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path

def main():
    # Run threshold analysis
    results = calculate_threshold_analysis()
    
    # Save results
    results_path = RESULTS_DIR / "threshold_analysis_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\n\nResults saved: {results_path}")
    
    # Create figure
    fig_path = create_threshold_figure(results)
    print(f"Figure saved: {fig_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: INTERVENTIONS SUFFICIENT TO CLOSE 15% OF MORTALITY GAP ALONE")
    print("=" * 80)
    
    sufficient = results[(results['target_gap_closure'] == '15%') & (results['sufficient_alone'] == True)]
    if len(sufficient) > 0:
        for _, row in sufficient.iterrows():
            print(f"  ✓ {row['intervention']}: {row['estimated_mortality_impact']:.1f} deaths/100k")
    else:
        print("  No single intervention is sufficient to close 15% alone")
    
    print("\n" + "=" * 80)
    print("POLICY IMPLICATIONS")
    print("=" * 80)
    print("""
1. HIGHEST IMPACT SINGLE INTERVENTIONS (GRADE-adjusted):
   - Clinical Pharmacist for Hypertension: ~13 deaths/100k (MODERATE)
   - Medicaid Expansion: ~11 deaths/100k (MODERATE)  
   - Clinical Pharmacist for Heart Failure: ~7 deaths/100k (HIGH)

2. COMBINED STRATEGY REQUIRED:
   - No single intervention closes >15% of gap
   - Portfolio approach needed: Coverage + Workforce + Technology
   
3. AI INTERVENTIONS:
   - Current evidence insufficient for mortality claims
   - AI Scribes: Indirect via workforce retention (~2 deaths/100k)
   - AI Chatbots: Safety concerns preclude mortality translation

4. Strongest evidence (HIGH GRADE) supports:
   - Integrated Behavioral Health for depression
   - Clinical Pharmacist integration for heart failure
""")
    
    return results

if __name__ == "__main__":
    results = main()
