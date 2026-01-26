# Primary Care Threshold Analysis

Code repository for: "Threshold Analysis of Primary Care Interventions and Mortality: Effect Benchmarks for Policy, Workforce, and Technology Approaches"

## Overview

This repository contains the analysis code to reproduce all results presented in the manuscript. The analysis integrates publicly available data sources to estimate associations between primary care interventions and mortality.

## Data Sources

All data used in this analysis are publicly available:

1. **NHANES Public-Use Linked Mortality Files**
   - Source: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
   - Individual-level mortality data with follow-up through December 2019

2. **CDC WONDER Compressed Mortality Files**
   - Source: https://wonder.cdc.gov/
   - State-level age-adjusted mortality rates (2014-2023)

3. **CDC Social Vulnerability Index**
   - Source: https://www.atsdr.cdc.gov/placeandhealth/svi/
   - Census tract-level vulnerability metrics

4. **Kaiser Family Foundation Medicaid Data**
   - Source: https://www.kff.org/medicaid/
   - State Medicaid expansion status and dates

## Repository Structure

```
primary_care_ai_futures/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_download_data.py    # Download public data sources
│   ├── 02_process_data.py     # Process and integrate data
│   └── 03_analysis.py         # Main analysis (DiD, threshold analysis)
├── data/
│   ├── raw/                   # Raw downloaded data (not tracked)
│   └── processed/             # Processed analysis files
├── results/                   # Analysis output
└── figures/                   # Generated figures
```

## Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproduction

To reproduce all analyses:

```bash
# 1. Download data
python scripts/01_download_data.py

# 2. Process data
python scripts/02_process_data.py

# 3. Run analysis
python scripts/03_analysis.py
```

Note: CDC WONDER data requires a manual query.

## License

MIT License
