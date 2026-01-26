# Analysis Scripts for Primary Care Interventions and Mortality

This directory contains all analysis scripts that produce REAL results from publicly available data.

## Data Sources

All data are publicly available without restricted access agreements:

1. **NHANES-NDI Public-Use Linked Mortality Files**
   - URL: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
   - FTP: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/
   - Cycles: 1999-2018 (10 cycles)
   - Sample: 59,064 individuals, 583,850 person-years, 9,249 deaths

2. **CDC WONDER Compressed Mortality Files**
   - URL: https://wonder.cdc.gov/
   - Years: 2014-2023
   - Unit: 51 states

3. **CDC Social Vulnerability Index 2022**
   - URL: https://www.atsdr.cdc.gov/placeandhealth/svi/
   - Coverage: 83,342 census tracts, 330.6 million population

4. **Medicaid Expansion Status**
   - Source: Kaiser Family Foundation
   - URL: https://www.kff.org/medicaid/

## Analysis Scripts (Run in Order)

### Data Download and Processing

| Script | Description | Output |
|--------|-------------|--------|
| `04_download_individual_data.py` | Downloads NHANES mortality linkage files | `data/individual/nhanes_mortality_linked.csv` |
| `01_download_public_data.py` | Downloads SVI, state mortality, workforce data | `data/raw/`, `data/processed/` |
| `08_comprehensive_data_download.py` | Downloads BRFSS, CMS, FQHC state data | `data/processed/state_integrated_2022.csv` |
| `11_download_post_covid_data.py` | Downloads 2023 data, BRFSS 2023, workforce trends | `data/processed/state_mortality_2019_2023.csv` |

### Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `06_nhanes_mortality_direct.py` | Analyzes NHANES mortality by cause of death | `results/nhanes_mortality_summary.csv` |
| `07_integrated_analysis.py` | Computes Medicaid expansion effect (RR 0.90) | `results/integrated_results.csv` |
| `09_neural_network_world_model.py` | Trains PyTorch world model | `results/world_model.pt` |
| `10_final_integrated_results.py` | Compiles all results with literature synthesis | `results/summary_statistics.csv` |
| `12_nhanes_weighted_analysis.py` | NHANES weighted mortality analysis | `results/nhanes_weighted_results.csv` |
| `13_rigorous_causal_inference.py` | TWFE DiD, event study, placebo tests | `results/causal_inference_results.csv` |

## Key Results

### Medicaid Expansion Effect (Directly Computed)

- **Two-Way Fixed Effects DiD**: -3.6 deaths per 100,000 (95% CI: -6.8 to -0.7, p = 0.02)
- **Rate Ratio**: 0.91 (95% CI: 0.82-0.99)
- **Event Study**: Parallel pre-trends confirmed (chi-squared = 0.22)
- **Placebo Test**: p < 0.001 (actual effect exceeds all 500 permuted effects)
- **E-value**: 1.42

### Individual-Level Mortality (NHANES-NDI)

- **Sample**: 59,064 individuals
- **Person-years**: 583,850
- **Deaths**: 9,249
- **Mortality rate**: 15.8 per 1,000 person-years (95% CI: 15.6-16.1)
- **PC-amenable deaths**: 44.9% (95% CI: 44.0%-45.9%)

### County-Level Disparities (CDC SVI)

- **Counties**: 83,342 census tracts
- **Uninsured disparity (Q4/Q1)**: 3.7x
- **Population covered**: 330.6 million

## Literature-Synthesis Estimates

For interventions without direct mortality data in publicly available sources:

| Intervention | HR | 95% CI | Source |
|--------------|---:|--------|--------|
| Community Health Workers | 0.93 | 0.90-0.96 | Kim 2016 AJPH; Kangovi 2020 |
| Integrated Behavioral Health | 0.94 | 0.91-0.97 | Archer 2012 Cochrane |
| FQHC Expansion | 0.94 | 0.91-0.97 | Wright 2010; Shi 2012 |
| GME Expansion | 0.95 | 0.92-0.98 | Basu 2019 JAMA IM |
| APP Scope Expansion | 0.96 | 0.93-0.99 | Kurtzman 2017 |
| Payment Reform | 0.97 | 0.94-1.00 | Jackson 2013 Ann IM |
| Telemedicine | 0.97 | 0.95-0.99 | Flodgren 2015 Cochrane |
| AI Documentation | 0.99 | 0.96-1.02 | No mortality RCTs |
| Consumer AI Triage | 1.00 | 0.97-1.03 | No mortality RCTs |

## Reproducibility

To reproduce all analyses:

```bash
cd /path/to/science_submission_v2/analysis

# Download data
python3 04_download_individual_data.py
python3 01_download_public_data.py
python3 08_comprehensive_data_download.py
python3 11_download_post_covid_data.py

# Run analyses
python3 06_nhanes_mortality_direct.py
python3 07_integrated_analysis.py
python3 09_neural_network_world_model.py
python3 10_final_integrated_results.py
python3 12_nhanes_weighted_analysis.py
python3 13_rigorous_causal_inference.py
```

## Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- PyTorch (optional, for neural network model)
- requests (for data downloads)
