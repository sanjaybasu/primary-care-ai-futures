# Data Directory

This directory contains documentation for accessing the data necessary to reproduce the analyses in Basu et al. (2026), "Primary Care Interventions and Mortality: Associations Estimated via Causal Inference Methods from Integrated National Data."

## Data Sources Overview

The analysis integrates seven national data sources:

| Data Source | Sample | Variables | Access |
|:---|:---|:---|:---|
| NHIS-NDI | 1,247,382 person-years | Individual mortality, demographics, access | Restricted |
| MEPS-NDI | 456,921 person-years | Utilization, expenditures, mortality | Restricted |
| NHANES | 78,246 adults | Clinical measurements | Public |
| NAMCS | 134,847 visits | Practice patterns | Public |
| AHRF | 3,143 counties | Workforce supply | Public |
| CDC WONDER | 3,143 counties | County mortality rates | Public |
| State Policies | 50 states + DC | Policy timing | Public |

## Directory Structure

```
data/
├── raw/                    # Instructions for downloading public data
│   ├── AHRF_download.md
│   ├── NHANES_download.md
│   ├── NAMCS_download.md
│   └── CDC_WONDER_download.md
├── processed/              # Pre-processed county-level aggregates
│   ├── county_svi_scores.csv
│   ├── state_policy_timing.csv
│   └── provider_ratios_2008_2023.csv
└── synthetic/              # Synthetic data for testing code
    └── synthetic_nhis_sample.csv
```

## Restricted Data Access

### NHIS-NDI Linkage (Primary Mortality Data)

**Source:** National Center for Health Statistics (NCHS)

**Application:** NCHS Research Data Center
- URL: https://www.cdc.gov/rdc/
- Process: Submit proposal, obtain IRB approval, execute data use agreement
- Timeline: 2-4 months

**Variables Used:**
- NHIS 2008-2023 survey responses (demographics, insurance, healthcare access)
- NDI mortality follow-up through December 2024
- County identifiers for linkage to AHRF/policy data

### MEPS-NDI Linkage (Validation Cohort)

**Source:** Agency for Healthcare Research and Quality (AHRQ)

**Application:** AHRQ Data Center
- URL: https://meps.ahrq.gov/
- Process: Submit proposal, IRB approval, restricted access agreement
- Timeline: 2-3 months

**Variables Used:**
- MEPS 2008-2023 (Panels 13-28)
- Healthcare utilization, expenditures, access measures
- NDI mortality linkage

## Public Data Sources

### NHANES (Clinical Measurements)

**Source:** CDC National Center for Health Statistics
- URL: https://wwwn.cdc.gov/nchs/nhanes/
- Format: SAS transport files, direct download
- Years: 2007-2023 cycles

**Variables:**
- Blood pressure, glucose, lipids
- BMI, physical examination data
- Laboratory results

### NAMCS (Practice Patterns)

**Source:** CDC National Center for Health Statistics
- URL: https://www.cdc.gov/nchs/ahcd/
- Format: SAS/Stata datasets
- Years: 2008-2021

**Variables:**
- Visit characteristics, diagnoses
- Telemedicine use, EHR adoption
- Time with provider

### Area Health Resources Files (AHRF)

**Source:** HRSA Data Warehouse
- URL: https://data.hrsa.gov/data/download
- Format: CSV/Excel
- Years: 2008-2023

**Variables:**
- Primary care physicians per 100,000
- Nurse practitioners, physician assistants
- HPSA designation scores

### CDC WONDER (County Mortality)

**Source:** CDC Wide-ranging Online Data for Epidemiologic Research
- URL: https://wonder.cdc.gov/
- Format: Tab-delimited text
- Years: 2008-2023

**Variables:**
- Age-adjusted mortality rates by county
- Cause-specific mortality (ICD-10)
- Population denominators

### State Policy Data

**Sources:** Kaiser Family Foundation, NCSL, individual state sources
- Medicaid expansion timing
- Scope-of-practice laws for APPs
- Telemedicine parity laws
- GME funding allocations

## Processed Data Files

For users without restricted data access, we provide county-level aggregates:

### `county_svi_scores.csv`
Social Vulnerability Index components by county-year (2008-2023)

### `state_policy_timing.csv`
Policy implementation dates:
- `medicaid_expansion_date`: Effective date of Medicaid expansion (if applicable)
- `app_scope_full_practice`: Year of full practice authority
- `telemedicine_parity_date`: Date of parity law enactment

### `provider_ratios_2008_2023.csv`
Workforce supply by county-year:
- `pcp_per_100k`: Primary care physicians per 100,000
- `np_per_100k`: Nurse practitioners per 100,000
- `pa_per_100k`: Physician assistants per 100,000

## Synthetic Data

For code testing without restricted data, `synthetic/synthetic_nhis_sample.csv` provides simulated individual-level data with similar statistical properties.

**Note:** Results from synthetic data will differ from manuscript values. Use restricted data for exact reproduction.

## Data Citations

1. National Center for Health Statistics. National Health Interview Survey, 2008-2023, linked to National Death Index through 2024. Hyattsville, MD: CDC/NCHS.

2. Agency for Healthcare Research and Quality. Medical Expenditure Panel Survey, 2008-2023, linked to National Death Index. Rockville, MD: AHRQ.

3. National Center for Health Statistics. National Health and Nutrition Examination Survey, 2007-2023. Hyattsville, MD: CDC/NCHS.

4. National Center for Health Statistics. National Ambulatory Medical Care Survey, 2008-2021. Hyattsville, MD: CDC/NCHS.

5. Health Resources and Services Administration. Area Health Resources Files, 2008-2023. Rockville, MD: HRSA.

6. Centers for Disease Control and Prevention. CDC WONDER Online Database. Atlanta, GA: CDC.
