#!/usr/bin/env python3
"""verify covid validation claims: R²=0.953 and MAPE=2.3%"""
import pandas as pd
import sys
from pathlib import Path

def verify_covid_r2():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "results" / "covid_validation_results.csv"

    if not results_file.exists():
        return {
            'claim': 'COVID validation R² = 0.953',
            'expected': 0.953,
            'actual': None,
            'passed': False,
            'error': f'file not found: {results_file}',
            'file': 'covid_validation_results.csv'
        }

    df = pd.read_csv(results_file)
    r2 = df['model_r2'].iloc[0]

    expected = 0.953
    tolerance = 0.001

    passed = abs(r2 - expected) < tolerance

    return {
        'claim': 'COVID validation R² = 0.953',
        'expected': expected,
        'actual': r2,
        'passed': passed,
        'file': 'covid_validation_results.csv'
    }

def verify_covid_mape():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "results" / "covid_validation_results.csv"

    if not results_file.exists():
        return {
            'claim': 'COVID validation MAPE = 2.3%',
            'expected': 2.3,
            'actual': None,
            'passed': False,
            'error': f'file not found: {results_file}',
            'file': 'covid_validation_results.csv'
        }

    df = pd.read_csv(results_file)
    mape = df['model_mape'].iloc[0]

    expected = 2.3
    tolerance = 0.1

    # round to 1 decimal for comparison
    mape_rounded = round(mape, 1)
    passed = abs(mape_rounded - expected) < tolerance

    return {
        'claim': 'COVID validation MAPE = 2.3%',
        'expected': expected,
        'actual': mape,
        'passed': passed,
        'file': 'covid_validation_results.csv',
        'details': f'actual: {mape:.2f}%, rounds to {mape_rounded}%'
    }

if __name__ == '__main__':
    results = [verify_covid_r2(), verify_covid_mape()]

    all_passed = True
    for result in results:
        if result['passed']:
            print(f"✓ PASS: {result['claim']}")
        else:
            print(f"✗ FAIL: {result['claim']}")
            all_passed = False

        print(f"  Expected: {result['expected']}")
        if result['actual'] is not None:
            if 'MAPE' in result['claim']:
                print(f"  Actual:   {result['actual']:.2f}")
            else:
                print(f"  Actual:   {result['actual']:.4f}")
        else:
            print(f"  Actual:   None")
        print(f"  File:     {result['file']}")
        if 'details' in result:
            print(f"  Details:  {result['details']}")
        if 'error' in result:
            print(f"  Error:    {result['error']}")
        print()

    sys.exit(0 if all_passed else 1)
