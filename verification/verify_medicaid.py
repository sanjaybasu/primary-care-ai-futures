#!/usr/bin/env python3
"""verify medicaid expansion claim: HR=0.90 (95% CI 0.82-0.99)"""
import pandas as pd
import sys
from pathlib import Path

def verify_medicaid():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "results" / "main_results.csv"

    if not results_file.exists():
        return {
            'claim': 'medicaid expansion HR = 0.90 (0.82-0.99)',
            'expected': '0.90 (0.82-0.99)',
            'actual': None,
            'passed': False,
            'error': f'file not found: {results_file}',
            'file': 'main_results.csv'
        }

    df = pd.read_csv(results_file)

    # extract values
    hr = float(df[df['Analysis'] == 'Mortality Rate Ratio (Expansion/Non-Expansion)']['Value'].iloc[0])
    ci_lower = float(df[df['Analysis'] == '95% CI Lower']['Value'].iloc[0])
    ci_upper = float(df[df['Analysis'] == '95% CI Upper']['Value'].iloc[0])

    expected_hr = 0.90
    expected_ci = (0.82, 0.99)

    # check with rounding tolerance
    hr_close = abs(hr - expected_hr) < 0.02
    ci_lower_close = abs(ci_lower - expected_ci[0]) < 0.02
    ci_upper_close = abs(ci_upper - expected_ci[1]) < 0.02

    passed = hr_close and ci_lower_close and ci_upper_close

    return {
        'claim': 'medicaid expansion HR = 0.90 (0.82-0.99)',
        'expected': f'{expected_hr} ({expected_ci[0]}-{expected_ci[1]})',
        'actual': f'{hr:.3f} ({ci_lower:.3f}-{ci_upper:.3f})',
        'passed': passed,
        'file': 'main_results.csv'
    }

if __name__ == '__main__':
    result = verify_medicaid()

    if result['passed']:
        print(f"✓ PASS: {result['claim']}")
    else:
        print(f"✗ FAIL: {result['claim']}")

    print(f"  Expected: {result['expected']}")
    print(f"  Actual:   {result['actual']}" if result['actual'] else f"  Actual:   None")
    print(f"  File:     {result['file']}")
    if 'error' in result:
        print(f"  Error:    {result['error']}")

    sys.exit(0 if result['passed'] else 1)
