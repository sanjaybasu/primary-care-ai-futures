# Downloading Area Health Resources Files (AHRF)

## Overview

The Area Health Resources Files contain county-level data on health facilities, health professions, health status, economic activity, and socioeconomic characteristics.

## Download Instructions

### Option 1: Direct Download (Recommended)

1. Go to: https://data.hrsa.gov/data/download
2. Navigate to "Area Health Resources Files"
3. Download the most recent complete file (usually includes historical data)
4. Extract to `data/raw/ahrf/`

### Option 2: Via Data Warehouse

1. Visit: https://data.hrsa.gov/topics/health-workforce/ahrf
2. Click "Download Data"
3. Select CSV format
4. Save to `data/raw/ahrf/`

## Required Variables

Our analysis uses:
- **County identifiers:** County FIPS, State FIPS
- **Primary care physicians:** Total PCPs per 100,000 population
- **Geographic:** Rural-Urban Continuum Codes
- **Years:** 2015-2025 (where available)

## File Structure

After download, you should have:
```
data/raw/ahrf/
├── ahrf2023.csv (or most recent year)
└── data_dictionary.xlsx
```

## Processing

Run the processing script:
```bash
python src/process_ahrf.py
```

This creates: `data/processed/provider_ratios_2015_2025.csv`

## Citation

Health Resources and Services Administration (HRSA). Area Health Resources Files. Rockville, MD: US Department of Health and Human Services.

## More Information

- Documentation: https://data.hrsa.gov/data/download
- User manual: Available on download page
- Contact: DataRequest@hrsa.gov
