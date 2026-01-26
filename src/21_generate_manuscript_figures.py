#!/usr/bin/env python3
"""
Generate Manuscript Figures
1. Causal Forest CATE Distribution (Violin Plot)
2. Intervention Effect Sizes (Forest Plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/primary_care_future_science_submission/science_submission_v2")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def plot_cate_distribution():
    print("Generating CATE violin plot...")
    df = pd.read_csv(RESULTS_DIR / "state_cate_estimates.csv")
    
    # Create quartiles if not present or recalculate to be sure
    # Using 'vulnerability_index' if available, else 'mean_svi'
    if 'vulnerability_index' in df.columns:
        metric = 'vulnerability_index'
    else:
        metric = 'mean_svi'
        
    df['SVI_Quartile'] = pd.qcut(df[metric], 4, labels=['Low (Q1)', 'Moderate-Low (Q2)', 'Moderate-High (Q3)', 'High (Q4)'])
    
    plt.figure(figsize=(10, 6))
    
    # Violin plot
    sns.violinplot(data=df, x='SVI_Quartile', y='cate_tlearner', palette='viridis', inner='quartile')
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel('Estimated Reduction in Deaths per 100,000\n(Negative = Benefit)')
    plt.xlabel('Social Vulnerability (SVI) Quartile')
    plt.title('Heterogeneity of Medicaid Expansion Effects by Community Vulnerability')
    
    # Annotate means
    means = df.groupby('SVI_Quartile')['cate_tlearner'].mean()
    for i, mean in enumerate(means):
        plt.text(i, mean, f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_heterogeneity.png", dpi=300)
    plt.close()
    print("Saved figure_4_heterogeneity.png")

def plot_intervention_forest():
    print("Generating Intervention Forest Plot...")
    
    # Manual data for the main figure based on our results
    # Combining Empirical (DiD), Meta-Analysis, and Literature
    
    interventions = [
        {'name': 'Medicaid Expansion', 'effect': 0.90, 'ci_low': 0.82, 'ci_high': 0.99, 'type': 'Policy', 'source': 'Empirical (DiD)'},
        {'name': 'Comm. Health Workers', 'effect': 0.93, 'ci_low': 0.90, 'ci_high': 0.96, 'type': 'Delivery', 'source': 'Meta-Analysis'},
        {'name': 'Int. Behavioral Health', 'effect': 0.94, 'ci_low': 0.91, 'ci_high': 0.97, 'type': 'Delivery', 'source': 'Meta-Analysis'},
        {'name': 'FQHC Expansion', 'effect': 0.94, 'ci_low': 0.91, 'ci_high': 0.97, 'type': 'Delivery', 'source': 'Lit Synthesis'},
        {'name': 'GME Expansion', 'effect': 0.95, 'ci_low': 0.92, 'ci_high': 0.98, 'type': 'Workforce', 'source': 'Lit Synthesis'},
        {'name': 'Telemedicine', 'effect': 0.97, 'ci_low': 0.95, 'ci_high': 0.99, 'type': 'Technology', 'source': 'Lit Synthesis'},
        {'name': 'AI Documentation', 'effect': 0.99, 'ci_low': 0.96, 'ci_high': 1.02, 'type': 'Technology', 'source': 'Extrapolation'},
        {'name': 'Consumer AI Triage', 'effect': 1.00, 'ci_low': 0.97, 'ci_high': 1.03, 'type': 'Technology', 'source': 'Extrapolation'},
    ]
    
    df = pd.DataFrame(interventions)
    
    plt.figure(figsize=(12, 8))
    
    # Colors by type
    colors = {'Policy': '#2ecc71', 'Delivery': '#3498db', 'Workforce': '#9b59b6', 'Technology': '#e74c3c'}
    
    # Plot
    y_pos = np.arange(len(df))
    plt.errorbar(x=df['effect'], y=y_pos, xerr=[df['effect']-df['ci_low'], df['ci_high']-df['effect']], 
                 fmt='o', color='black', capsize=5, zorder=3)
    
    for i, row in df.iterrows():
        plt.barh(i, width=0, color=colors[row['type']], label=row['type'] if row['type'] not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(row['effect'], i, 'o', color=colors[row['type']], markersize=10, zorder=4)
        
        # Text annotation
        plt.text(row['ci_high'] + 0.01, i, f"{row['effect']:.2f} ({row['ci_low']:.2f}-{row['ci_high']:.2f})", va='center')

    plt.yticks(y_pos, df['name'])
    plt.axvline(1.0, color='gray', linestyle='--', alpha=0.8)
    plt.xlabel('Mortality Rate Ratio (Hazard Ratio)')
    plt.title('Comparative Causal Effects of Primary Care Interventions on Mortality')
    
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.xlim(0.8, 1.1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_3_forest_plot.png", dpi=300)
    plt.close()
    print("Saved figure_3_forest_plot.png")

if __name__ == "__main__":
    plot_cate_distribution()
    plot_intervention_forest()
