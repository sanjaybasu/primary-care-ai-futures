# Data Directory

This directory contains the data necessary to reproduce the analyses in Basu et al. (2026).

## Directory Structure

```
data/
├── raw/                    # Links and instructions for downloading public data
│   ├── AHRF_download.md
│   ├── CHR_download.md
│   ├── MEPS_download.md
│   └── ACS_download.md
├── processed/              # Pre-processed county-level aggregates
│   ├── county_svi_scores.csv
│   ├── digital_readiness_2025.csv
│   └── provider_ratios_2015_2025.csv
└── synthetic/              # Synthetic data for testing code
    └── synthetic_meps_sample.csv
```

## Public Data Sources

All primary data sources are publicly available. Due to data use agreements, we cannot redistribute raw individual-level MEPS data, but we provide:

1. **Download instructions** in `raw/` subdirectory
2. **Processed county-level aggregates** (non-identifiable) in `processed/`
3. **Synthetic data** matching statistical properties for code testing in `synthetic/`

### Area Health Resources Files (AHRF)
- **Source:** HRSA Data Warehouse
- **URL:** https://data.hrsa.gov/data/download
- **Years:** 2015-2025
- **Variables:** Primary care physicians per 100,000 population, HPSA scores
- **Format:** CSV download
- **Instructions:** See `raw/AHRF_download.md`

### County Health Rankings (CHR)
- **Source:** University of Wisconsin Population Health Institute
- **URL:** https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation
- **Years:** 2015-2025
- **Variables:** Premature death (YPLL), poverty, unemployment, housing burden
- **Format:** CSV download
- **Instructions:** See `raw/CHR_download.md`

### Medical Expenditure Panel Survey (MEPS)
- **Source:** AHRQ MEPS
- **URL:** https://meps.ahrq.gov/mepsweb/
- **Years:** 2015-2022 (Panels 20-27)
- **Variables:** Healthcare utilization, demographics, insurance, chronic conditions
- **Format:** Stata (.dta) files
- **Restriction:** Data Use Agreement required
- **Instructions:** See `raw/MEPS_download.md` for detailed steps

### American Community Survey (ACS)
- **Source:** US Census Bureau
- **URL:** https://www.census.gov/programs-surveys/acs/data.html
- **Product:** 5-Year Estimates (2021)
- **Variables:** Age distribution, broadband penetration, household income
- **Format:** CSV download via data.census.gov
- **Instructions:** See `raw/ACS_download.md`

## Processed Data

County-level aggregates are provided in `processed/` and can be used directly:

### `county_svi_scores.csv`
Columns:
- `fips`: County FIPS code
- `year`: 2015-2025
- `poverty_rate`: Percentage below poverty line
- `unemployment_rate`: Unemployment percentage
- `housing_burden`: % renter-occupied with >30% cost burden
- `svi_proxy`: Standardized composite (z-score average)
- `svi_quartile`: 1 (lowest) to 4 (highest vulnerability)

### `digital_readiness_2025.csv`
Columns:
- `fips`: County FIPS code
- `broadband_penetration`: Fraction with >25/3 Mbps
- `pct_age_18_34`: Population fraction aged 18-34
- `pct_age_35_64`: Population fraction aged 35-64
- `pct_age_65plus`: Population fraction aged 65+
- `digital_readiness`: Computed score (0-1)

### `provider_ratios_2015_2025.csv`
Columns:
- `fips`: County FIPS code
- `year`: 2015-2025
- `pcp_per_100k`: Primary care physicians per 100,000 population
- `population`: County population
- `hpsa_score`: Health Professional Shortage Area score

## Synthetic Data

For users who cannot access MEPS due to data use restrictions, we provide synthetic individual-level data in `synthetic/synthetic_meps_sample.csv` that preserves the statistical structure for code testing.

**Note:** Synthetic data will produce similar but not identical results to those in the manuscript. For exact reproduction, use the actual MEPS data following instructions in `raw/MEPS_download.md`.

## Data Citations

1. Health Resources and Services Administration (HRSA). Area Health Resources Files, 2015-2025. Rockville, MD: US Department of Health and Human Services.

2. University of Wisconsin Population Health Institute. County Health Rankings \u0026 Roadmaps, 2015-2025. Madison, WI.

3. Agency for Healthcare Research and Quality (AHRQ). Medical Expenditure Panel Survey, 2015-2022. Rockville, MD: US Department of Health and Human Services.

4. US Census Bureau. American Community Survey 5-Year Estimates, 2021. Washington, DC: US Department of Commerce.
