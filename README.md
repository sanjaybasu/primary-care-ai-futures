# Strategies and Thresholds to Close the Primary Care Mortality Gap

Code and data repository for: **"Strategies and Thresholds to Close the Primary Care Mortality Gap: A Calibrated Causal Simulation"**

**Authors**: Sanjay Basu, Ishani Ganguli, Asaf Bitton, Robert L. Phillips Jr., Zirui Song, Russell Scott Phillips, Bruce E. Landon

---

## Overview

This repository contains analysis code, data processing pipelines, and reproduction scripts. We integrated individual-level clinical data (NHANES-NDI), county-level infrastructure data (AHRF, SVI), and meta-analyses of 91 studies to model the mortality effects of traditional policy interventions versus emerging artificial intelligence technologies.


---

## Repository Structure

```
primary_care_ai_futures/
├── README.md                          # this file
├── requirements.txt                   # python dependencies
├── src/                               # detailed analysis modules (27 files)
│   ├── 17_formal_dml.py              # double machine learning
│   ├── 09_neural_network_world_model.py  # RSSM simulation
│   ├── 23_iv_analysis.py             # instrumental variables
│   └── [...]                         # additional analysis scripts
├── data/
│   ├── raw/                          # raw data (requires download)
│   └── processed/                    # integrated datasets
├── results/                          # analysis outputs
```

---

## Methods

### Three-Stage Architecture

1. **Data Integration**: 1.9M person-year observations from NHANES-NDI, NHIS, MEPS linked to county-level AHRF and CDC SVI data

2. **Causal Estimation**:
   - Double Machine Learning (DML) with 5-fold cross-fitting
   - Instrumental Variables (IV) using residency density, Anderson-Rubin robust inference
   - Callaway-Sant'Anna difference-in-differences for Medicaid expansion

3. **Simulation Engine**: Recurrent State-Space Model (RSSM) calibrated on 2014-2019 data, validated against:
   - COVID-19 pandemic (R²=0.953)
   - Medicaid expansion policy variation (MAPE 0.25%)
   - 2020 telemedicine surge (r=0.84 access restoration)

---

## Installation

### Requirements
- Python 3.8+
- Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- Statistics: `statsmodels`, `econml`
- Optional: `torch` (for neural net implementation; has sklearn fallback)

```bash
pip install -r requirements.txt
```

---

## Data Requirements

### Public Data (Downloadable)
- **NHANES Mortality Files**: CDC/NCHS public-use linked mortality (1999-2018)
- **CDC SVI**: Social Vulnerability Index 2022
- **Medicaid Expansion**: State adoption dates (included in repo)

### Manual Download Required
- **CDC WONDER**: State-level mortality rates 2014-2023 (requires data use agreement)
- **NHANES-NDI**: Full linkage files (requires RDC approval for restricted variables)

### Missing from Repository
Due to size and data use agreements, the following are NOT included:
- `data/raw/NHANES_*_MORT_2019_PUBLIC.dat` (10 files)
- `data/raw/wonder_state_mortality.txt`
- `data/individual/nhanes_mortality_linked.csv`
- `data/processed/state_integrated_2022.csv`

See `REPRODUCTION_AUDIT.md` for complete data inventory.

---

## Reproduction

```bash
# download public data
python scripts/01_download_data.py

# process available data
python scripts/02_process_data.py

# run analysis (will work with synthetic data for testing)
python scripts/03_analysis.py
```


---


## Citation

If using this code or data, please cite:

```
Basu S, Ganguli I, Bitton A, Phillips RL Jr, Song Z, Phillips RS, Landon BE.
Strategies and Thresholds to Close the Primary Care Mortality Gap: A Calibrated Causal Simulation.
[Journal] [Year]. [DOI]
```

---

## License

MIT License

---

## Contact

**Corresponding Author**: Sanjay Basu, MD PhD
Email: sanjay.basu@ucsf.edu
Affiliation: University of California San Francisco & Waymark
