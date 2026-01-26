# Strategies and Thresholds to Close the Primary Care Mortality Gap: A Calibrated Causal Simulation


## Overview

This repository contains the analysis code, data processing pipelines, and reproduction scripts for the study. 

## Repository Structure

```
primary_care_ai_futures/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── 01_download_data.py    # Download public data (NHANES, AHRF)
│   ├── 02_process_data.py     # Data integration and preprocessing
│   └── 03_analysis.py         # Main analysis (DiD, DML, RSSM Simulation)
├── data/
│   ├── raw/                   # Raw input data (not tracked)
│   └── processed/             # Integrated 'Universal Corpus'
```

## Methods

The analysis uses a **Calibrated Policy Simulation** approach:
1.  **Causal Parameters:** Estimated using Double Machine Learning (DML) for high-dimensional adjustment and Instrumental Variables (IV) with Anderson-Rubin robust inference.
2.  **Simulation Engine:** A Recurrent State-Space Model (RSSM) trained on 15 years of system dynamics to project counterfactual outcomes.
3.  **Validation:** Validated against three natural experiments: COVID-19 resilience ($$R^2=0.95$$), Medicaid expansion (Policy), and the 2020 telemedicine shock (Substitution).

## Requirements

*   Python 3.8+
*   `pandas`, `numpy`, `scipy`, `scikit-learn`
*   `econml` (for Double Machine Learning)
*   `statsmodels` (for Time Series/IV)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproduction

To reproduce the full analysis pipeline:

```bash
# 1. Download public datasets
python scripts/01_download_data.py

# 2. Process and integrate into Universal Corpus
python scripts/02_process_data.py

# 3. Run analysis and generate figures
python scripts/03_analysis.py
```

*Note: Access to CDC WONDER detailed mortality files requires a user agreement and cannot be automated; place downloaded files in `data/raw/wonder/`.*

## License

MIT License
