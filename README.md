# Primary Care AI Futures: Simulation Model and Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CodeOcean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/XXXXXXX)

**Associated Manuscript:** Basu S, Song Z, Phillips RL Jr, et al. Artificial Intelligence and the Future of Primary Care: Social Vulnerability Determines Impact. 2026.
---

## Overview

This repository contains the complete code and data to reproduce the analyses in Basu et al. (2026), which examines how the deployment modality of artificial intelligence in primary care affects access disparities.

**Key Findings:**
- Market-driven AI deployment exacerbates inequality (Q4:Q1 wait-time ratio increases from 1.8 to 2.4)
- Equity-focused deployment stabilizes disparities at ICER of $68,000/QALY
- Temporal validation confirms model predictive validity (AUC 0.81, calibration slope 0.95)

---

## Repository Structure

```
├── data/
│   ├── processed/                # Pre-processed county-level data (synthetic)
│   ├── raw/                      # Links to public data sources
│   └── README.md                 # Data dictionary and sources
├── src/
│   ├── rssm_architecture.py      # Core world model implementation
│   ├── rssm_training.py          # Model training script
│   ├── scenario_simulation.py    # Future scenario simulation
│   ├── cost_effectiveness.py     # CEA analysis
│   └── figure_generation.py      # Publication figures
├── notebooks/
│   ├── 01_comprehensive_trends.ipynb
│   ├── 02_rssm_science_sim.ipynb
│   ├── 03_temporal_validation.ipynb
│   └── 04_cost_effectiveness.ipynb
├── results/
│   ├── figures/                  # Publication-quality figures
│   ├── tables/                   # CSV tables for manuscript
│   └── validation/               # Model validation outputs
├── docs/
│   ├── TRIPOD_AI_checklist.md
│   ├── PRISMA_flowchart.pdf
│   └── methods_supplement.pdf
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md (this file)
```

---

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip

### Setup

```bash
# Clone repository
git clone https://github.com/sanjaybasu/primary-care-ai-futures.git
cd primary-care-ai-futures

# Create conda environment
conda env create -f environment.yml
conda activate pc-ai-futures

# OR use pip
pip install -r requirements.txt
```

---

## Reproduction Instructions

### 1. Data Preparation

Public data sources used:
- **AHRF** (Area Health Resources Files): [HRSA Data Warehouse](https://data.hrsa.gov/data/download)
- **CHR** (County Health Rankings): [CHR \u0026 Roadmaps](https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation)
- **MEPS** (Medical Expenditure Panel Survey): [AHRQ MEPS](https://meps.ahrq.gov/mepsweb/)
- **ACS** (American Community Survey): [Census Bureau](https://www.census.gov/programs-surveys/acs/data.html)

**Note:** Due to data use agreements, raw MEPS individual-level data cannot be redistributed. We provide:
1. Synthetic data matching the statistical properties for code testing
2. Download scripts and data dictionaries in `data/raw/`
3. Processed county-level aggregates (non-identifiable)

```bash
cd data/raw
python download_public_data.py  # Downloads AHRF, CHR, ACS
# Follow MEPS instructions in data/raw/MEPS_README.md
```

### 2. Run Analyses

Execute notebooks in order:

```bash
jupyter notebook notebooks/01_comprehensive_trends.ipynb
# Generates Figure 1 (Trends by SVI)

jupyter notebook notebooks/02_rssm_science_sim.ipynb  
# Generates Figure 2 (Scenario simulations)

jupyter notebook notebooks/03_temporal_validation.ipynb
# Validates model (Table 1, Supplementary Figure S1)

jupyter notebook notebooks/04_cost_effectiveness.ipynb
# CEA analysis (Table 2, Figure 3)
```

**OR** run complete pipeline:

```bash
python src/run_full_analysis.py --config config/science_submission.yaml
```

### 3. Verify Outputs

Expected results (with synthetic data):
- **Temporal Validation AUC:** 0.81 ± 0.03
- **Baseline→Market ICER:** $68,000/QALY ± $18,000
- **Q4:Q1 Wait Ratio (Equitable, 2035):** 1.7 ± 0.2

Compare your outputs to `results/expected_outputs.json`:

```bash
python tests/verify_reproduction.py
```

---

## Model Description

### Recurrent State-Space Model (RSSM)

The core simulation engine is a hierarchical RSSM that learns:
1. **Individual dynamics:** Patient utilization trajectories over time
2. **System dynamics:** County-level capacity constraints and demand

Architecture:
- Input: 15 individual features + 10 system features
- Latent dimensions: 32 (individual) + 16 (system)
- Training: Variational inference (ELBO objective)
- Validation: Temporal split (2015-2020 train, 2021-2025 test)

See `src/rssm_architecture.py` for implementation and `docs/methods_supplement.pdf` for mathematical details.

### AI Adoption Model

Digital readiness = $B \times (0.43 \times P_{18-34} + 0.20 \times P_{35-64} + 0.06 \times P_{65+})$

Where:
- $B$ = broadband penetration  
- $P_g$ = fraction of population in age group $g$
- Weights from systematic review (n=8 studies, 47,200 respondents)

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{basu2026primary,
  title={Artificial Intelligence and the Future of Primary Care: Social Vulnerability Determines Impact},
  author={Basu, Sanjay and Phillips Jr., Robert L. and Bitton, Asaf and Landon, Bruce E. and Song, Zirui and Phillips, Russell S.},
  journal={Science},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={10.1126/science.XXX}
}
```

---

## License

MIT License. See `LICENSE` for details.

The code is provided "as-is" for research purposes. Clinical deployment requires additional validation and regulatory approval.

---

## Contact

**Corresponding Author:** Sanjay Basu, MD PhD  
Email: sbasu@ucsf.edu  
Affiliation: Center for Primary Care, UCSF

**Issues \u0026 Pull Requests:** Welcome! Please open an issue for bugs or questions.

---

## Acknowledgments

- AHRQ for MEPS data access
- HRSA for AHRF data
- University of Wisconsin Population Health Institute for CHR data  
- CDC/ATSDR for SVI documentation
