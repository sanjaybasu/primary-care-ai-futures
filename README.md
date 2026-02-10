# Strategies and Thresholds to Close the Primary Care Mortality Gap

Code and data repository for: **"Strategies and Thresholds to Close the Primary Care Mortality Gap: A Calibrated Causal Simulation"**

**Authors**: Sanjay Basu, Ishani Ganguli, Asaf Bitton, Robert L. Phillips Jr., Zirui Song, Russell Scott Phillips, Bruce E. Landon

---

## Overview

This repository contains analysis code, data processing pipelines, and reproduction scripts. We integrated individual-level clinical data (NHANES-NDI), county-level infrastructure data (AHRF, SVI), and meta-analyses of 91 studies to model the mortality effects of traditional policy interventions versus emerging artificial intelligence technologies.

### Key Findings

- **Mortality Gap**: 2.4 deaths/1,000 person-years between high- and low-vulnerability communities, associated with 38% decline in physician supply in high-vulnerability areas
- **Effective Strategies**: Combined package (Medicaid expansion + CHW programs at 65% coverage + FQHC expansion + physician workforce + integrated behavioral health) could close **75% of the gap** (95% CI 62-88%)
- **AI Thresholds**: Current AI documentation tools show null mortality benefit (RR=0.99). For AI to match Medicaid's impact, it requires **nine-fold efficacy increase** (target RR=0.91)
- **Equity Requirement**: AI interventions must achieve equitable distribution across socioeconomic strata—likely requiring pairing with CHWs—to close mortality gaps

---

## Repository Structure

```
primary_care_ai_futures/
├── README.md                          # this file
├── CHANGES_SUMMARY.md                 # comprehensive revision log
├── REPRODUCTION_AUDIT.md              # reproducibility assessment
├── COAUTHOR_COMMENTS.md               # co-author feedback catalog
├── MANUSCRIPT_FINAL.md                # final revised manuscript
├── requirements.txt                   # python dependencies
├── scripts/                           # simplified analysis pipeline
│   ├── 01_download_data.py           # download NHANES, SVI data
│   ├── 02_process_data.py            # data integration
│   ├── 03_analysis.py                # main orchestrator
│   └── 20_world_model_threshold.py   # threshold analysis
├── src/                               # detailed analysis modules (27 files)
│   ├── 17_formal_dml.py              # double machine learning
│   ├── 09_neural_network_world_model.py  # RSSM simulation
│   ├── 23_iv_analysis.py             # instrumental variables
│   └── [...]                         # additional analysis scripts
├── data/
│   ├── raw/                          # raw data (requires download)
│   └── processed/                    # integrated datasets
├── results/                          # analysis outputs
└── figures/                          # manuscript figures
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

### Basic Reproduction (without restricted data)
```bash
# download public data
python scripts/01_download_data.py

# process available data
python scripts/02_process_data.py

# run analysis (will work with synthetic data for testing)
python scripts/03_analysis.py
```

### Full Reproduction (with CDC data access)
1. Obtain CDC WONDER mortality data (requires data use agreement)
2. Obtain NHANES-NDI linkage files (public use or restricted)
3. Place files in `data/raw/` as specified in scripts
4. Run full pipeline as above

### Key Analysis Scripts
- `src/17_formal_dml.py` - DML estimation of intervention effects
- `src/09_neural_network_world_model.py` - RSSM training and validation
- `src/23_iv_analysis.py` - IV analysis with Anderson-Rubin inference
- `src/18_covid_validation.py` - COVID-19 validation
- `scripts/20_world_model_threshold.py` - threshold analysis for gap closure

---

## Code Quality Notes

### Recent Improvements (2026-02-09)
- ✅ Fixed all hardcoded paths (now uses relative paths)
- ✅ De-AI'd codebase (concise comments, no superlatives)
- ✅ Cleaned docstrings (pithy, lowercase, technical)
- ✅ Removed verbose formatting

### Known Issues
- Dual codebase: `scripts/` (simplified 3-file pipeline) vs `src/` (detailed 27-file modules)
- Some `src/` files are development iterations; canonical versions in `scripts/`
- Import structure mixes module imports and subprocess calls

See `REPRODUCTION_AUDIT.md` for complete assessment.

---

## Key Limitations

### Geographic Unit
- Analysis uses **county-level** data, not Primary Care Service Areas (PCSAs)
- PCSAs may better align with care-seeking behavior and show larger effects

### Data Sources
- AHRF may overestimate primary care supply vs recent IQVIA estimates
- CDC SVI chosen for policy relevance; READI may capture different dimensions

### AI Equity Assumptions
- Model assumes **uniform AI benefit** across socioeconomic strata
- Real-world deployment likely exhibits differential uptake (digital divide)
- Achieving equity gap closure requires **AI + CHW pairing**, not AI substitution
- See `RESPONSE_TO_RUSS_PHILLIPS.md` for detailed equity discussion

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

---

## Change Log

**2026-02-09**: Major revision
- Fixed hardcoded paths across 9 files
- De-AI'd codebase (5 key files)
- Updated manuscript with all co-author feedback
- Added equity assumption limitations
- Corrected NASEM citation
- Expanded limitations (PCSA, AHRF, SVI, equity)
- See `CHANGES_SUMMARY.md` for complete revision log

**2024-01-26**: Initial public release
