#!/usr/bin/env python3
"""verify mortality gap claim: 2.4 deaths per 1,000 person-years"""
import pandas as pd
import sys
from pathlib import Path

def verify_mortality_gap():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "results" / "county_disparity_results.csv"

    if not results_file.exists():
        return {
            'claim': 'mortality gap = 2.4 deaths/1000',
            'expected': 2.4,
            'actual': None,
            'passed': False,
            'error': f'file not found: {results_file}',
            'file': 'county_disparity_results.csv'
        }

    df = pd.read_csv(results_file)

    # extract values
    q4_mortality = float(df[df['Metric'] == 'Mortality Q4 (per 1000)']['Value'].iloc[0])
    q1_mortality = float(df[df['Metric'] == 'Mortality Q1 (per 1000)']['Value'].iloc[0])
    gap = q4_mortality - q1_mortality

    expected = 2.4
    tolerance = 0.1

    passed = abs(gap - expected) < tolerance

    return {
        'claim': 'mortality gap = 2.4 deaths/1000',
        'expected': expected,
        'actual': gap,
        'passed': passed,
        'file': 'county_disparity_results.csv',
        'details': f'Q4: {q4_mortality}, Q1: {q1_mortality}'
    }

if __name__ == '__main__':
    result = verify_mortality_gap()

    if result['passed']:
        print(f"✓ PASS: {result['claim']}")
    else:
        print(f"✗ FAIL: {result['claim']}")

    print(f"  Expected: {result['expected']}")
    print(f"  Actual:   {result['actual']:.2f}" if result['actual'] else f"  Actual:   None")
    print(f"  File:     {result['file']}")
    if 'details' in result:
        print(f"  Details:  {result['details']}")
    if 'error' in result:
        print(f"  Error:    {result['error']}")

    sys.exit(0 if result['passed'] else 1)
