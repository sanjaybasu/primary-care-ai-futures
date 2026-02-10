#!/usr/bin/env python3
"""master verification script - runs all claim checks"""
import sys
from pathlib import Path

# import verification functions
sys.path.insert(0, str(Path(__file__).parent))
from verify_mortality_gap import verify_mortality_gap
from verify_medicaid import verify_medicaid
from verify_covid import verify_covid_r2, verify_covid_mape

def run_all_verifications():
    verifications = [
        verify_mortality_gap(),
        verify_medicaid(),
        verify_covid_r2(),
        verify_covid_mape(),
    ]

    print("=" * 70)
    print("MANUSCRIPT CLAIMS VERIFICATION REPORT")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for v in verifications:
        status = "✓ PASS" if v['passed'] else "✗ FAIL"
        print(f"{status}: {v['claim']}")
        print(f"  Expected: {v['expected']}")

        if v['actual'] is not None:
            if isinstance(v['actual'], float):
                print(f"  Actual:   {v['actual']:.4f}")
            else:
                print(f"  Actual:   {v['actual']}")
        else:
            print(f"  Actual:   None")

        print(f"  File:     {v['file']}")

        if 'details' in v:
            print(f"  Details:  {v['details']}")
        if 'error' in v:
            print(f"  Error:    {v['error']}")

        print()

        if v['passed']:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} claims")
    print("=" * 70)
    print()

    print("NOTE: This verifies 4 fully verified claims from results files.")
    print("Additional claims require further investigation:")
    print("  - workforce decline Q4 38% vs Q1 15%")
    print("  - telemedicine correlation r=0.84")
    print("  - Anderson-Rubin CI [-7.5, -1.8]")
    print("  - 75% gap closure (62-88%)")
    print("  - 9-fold AI improvement threshold")
    print()

    return passed, failed

if __name__ == '__main__':
    passed, failed = run_all_verifications()
    sys.exit(0 if failed == 0 else 1)
