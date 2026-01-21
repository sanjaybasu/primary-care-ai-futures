# Downloading MEPS Data

The Medical Expenditure Panel Survey (MEPS) is maintained by the Agency for Healthcare Research and Quality (AHRQ). Due to data use agreements, we cannot redistribute the raw microdata, but it is freely available for download.

## Required Files

You need the following MEPS Full Year Consolidated files (Stata format):

- **h216.dta** (2018 Full Year)
- **h209.dta** (2017 Full Year)  
- **h201.dta** (2016 Full Year)
- **h192.dta** (2015 Full Year)
- **h224.dta** (2019 Full Year)
- **h233.dta** (2020 Full Year)
- **h243.dta** (2021 Full Year - if available)
- **h250.dta** (2022 Full Year - if available)

## Download Instructions

### Step 1: Navigate to MEPS Website

Go to: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp

### Step 2: Download Each Year

For each year:

1. Click on the "Full Year Consolidated Data Files" section
2. Find the year you need (e.g., 2018, 2019, 2020, etc.)
3. Download the **Stata** version (`.dta` format)
4. File names follow the pattern `hXXX.dta` where XXX is the file number

Example URLs (check website for latest):
- 2018: https://meps.ahrq.gov/mepsweb/data_stats/download_data_file_detail.jsp?cboPufNumber=HC-216
- 2019: https://meps.ahrq.gov/mepsweb/data_stats/download_data_file_detail.jsp?cboPufNumber=HC-224
- 2020: https://meps.ahrq.gov/mepsweb/data_stats/download_data_file_detail.jsp?cboPufNumber=HC-233

### Step 3: Place Files

After downloading, place all `.dta` files in:
```
packaging/primary_care_ai_futures/data/raw/meps/
```

### Step 4: Run Data Preparation

Once files are in place, run:
```bash
cd packaging/primary_care_ai_futures
python src/rssm_data_loader.py
```

This will create `rssm_meps_prepared.csv` in the `data/processed/` directory.

## Alternative: Use Synthetic Data

If you cannot access MEPS data, we provide synthetic data that matches the statistical properties:

```bash
# Synthetic data is already provided in:
data/synthetic/synthetic_meps_sample.csv
```

**Note:** Synthetic data will produce similar but not identical results to the manuscript. For exact reproduction, use actual MEPS data.

## Data Use Agreement

MEPS data is public but subject to a data use agreement. By downloading MEPS data, you agree to:
1. Use the data only for statistical reporting and analysis
2. Make no attempt to identify survey participants
3. Not redistribute the microdata files

Full agreement: https://meps.ahrq.gov/data_stats/download_data/pufs/data_use.jsp

## Questions?

For MEPS download issues, contact AHRQ at MEPSProjectDirector@ahrq.hhs.gov
