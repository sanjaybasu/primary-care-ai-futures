# CodeOcean Reproducibility Capsule

This directory contains the configuration files for creating a CodeOcean computational reproducibility capsule.

## What is CodeOcean?

CodeOcean (https://codeocean.com) is a cloud-based computational reproducibility platform that allows researchers to package code, data, and computing environment together for complete reproducibility.

## Files in This Directory

- `environment.yml` - Conda environment specification
- `run_capsule.sh` - Main script that executes the full analysis pipeline
- `metadata.json` - CodeOcean metadata
- `README_CODEOCEAN.md` - Instructions for the capsule

## Creating the Capsule

### Option 1: Upload to CodeOcean

1. Create an account at https://codeocean.com
2. Click "New Capsule" → "Upload from GitHub"
3. Enter repository URL: `https://github.com/sanjaybasu/primary_care_ai_futures`
4. CodeOcean will automatically detect the environment and run script

### Option 2: Manual Setup

1. Create new capsule on CodeOcean
2. Upload all files from this repository
3. Set environment: Python 3.10
4. Copy `run_capsule.sh` as the run script
5. Upload data files to `/data` folder (or use synthetic data)

## Expected Runtime

- **With synthetic data:** ~30 minutes
- **With real MEPS data:** ~2-3 hours (depending on compute allocation)

## Output Files

After successful run, the capsule will generate:
- `results/figures/` - All 8 publication figures (3 main + 5 supplementary)
- `results/tables/` - CSV tables for manuscript
- `results/validation/` - Temporal validation metrics
- `results/cost_effectiveness/` - CEA outputs

## Computational Requirements

- **RAM:** 16 GB minimum, 32 GB recommended
- **CPU:** 4+ cores recommended
- **Storage:** 10 GB for data + outputs
- **GPU:** Optional (speeds up RSSM training 3-5x)

## Verification

Compare your outputs to `/docs/expected_outputs.json` to verify correct reproduction. Values should be within ±10% due to random seed variation in Monte Carlo simulations.

## Citation

If you use this capsule, please cite both the manuscript and the capsule DOI:

```bibtex
@misc{basu2026capsule,
  title={Computational Reproducibility Capsule: Primary Care AI Futures},
  author={Basu, Sanjay and Song, Zirui and Phillips Jr., Robert L. and Ganguli, Ishani and Landon, Bruce E. and Bitton, Asaf and Phillips, Russell S.},
  year={2026},
  publisher={CodeOcean},
  doi={10.24433/CO.XXXXXXX.v1}
}
```

## Support

For issues with the capsule:
- GitHub Issues: https://github.com/sanjaybasu/primary_care_ai_futures/issues
- Email: sbasu@ucsf.edu
