# Documentation

This directory contains supplementary documentation for the Basu et al. (2026) analyses.

## Contents

- **TRIPOD_AI_checklist.md** - Complete TRIPOD+AI reporting checklist (27 items)
- **CHEERS_checklist.md** - CHEERS 2022 economic evaluation checklist (25 items)
- **methods_supplement.pdf** - Full mathematical formulations and technical details
- **expected_outputs.json** - Expected numerical results for validation

## Reporting Guidelines Compliance

### TRIPOD+AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + AI)

**Citation:** Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378.

**Purpose:** Ensures transparent reporting of AI/ML prediction models.

**Status:** 100% compliance verified (see TRIPOD_AI_checklist.md)

### CHEERS 2022 (Consolidated Health Economic Evaluation Reporting Standards)

**Citation:** Husereau D, Drummond M, Augustovski F, et al. Consolidated Health Economic Evaluation Reporting Standards 2022 (CHEERS 2022) statement: updated reporting guidance for health economic evaluations. Value Health 2022;25:3-9.

**Purpose:** Standardized reporting for cost-effectiveness analyses.

**Status:** 100% compliance verified (see CHEERS_checklist.md)

### PRISMA 2020 (Preferred Reporting Items for Systematic Reviews and Meta-Analyses)

**Citation:** Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 2021;372:n71.

**Purpose:** Transparent reporting of systematic reviews (AI adoption parameters).

**Status:** Full flowchart and checklist in Supplementary Materials (appendix_revised.md)

## Expected Outputs

For validation of code reproduction, see `expected_outputs.json`:

```json
{
  "temporal_validation": {
    "auc_roc": 0.81,
    "auc_95ci_lower": 0.78,
    "auc_95ci_upper": 0.84,
    "brier_score": 0.102,
    "calibration_slope": 0.95
  },
  "cost_effectiveness": {
    "icer_equitable_vs_market": 68000,
    "icer_95ci_lower": 52000,
    "icer_95ci_upper": 89000,
    "prob_cost_effective_100k": 0.78
  },
  "simulation_2035": {
    "baseline_q4q1_ratio": 2.1,
    "market_ai_q4q1_ratio": 2.4,
    "equitable_ai_q4q1_ratio": 1.7
  }
}
```

If your reproduced values fall within ±10% of these benchmarks (accounting for random seed variation), reproduction is successful.

## Contact for Questions

For questions about methods or reproduction:
- **Email:** sbasu@ucsf.edu
- **GitHub Issues:** https://github.com/sanjaybasu/primary-care-ai-futures/issues
