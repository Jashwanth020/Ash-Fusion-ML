"""
pipeline_DT.py — Deformation Temperature (DT) Prediction Pipeline

Predicts DT (deformation temperature) of ash from coal and biomass feedstocks
using 6 tree-based ML models with Bayesian hyperparameter optimization.

Usage:
    python pipeline_DT.py              # Full mode (50 Optuna trials, 5×3 CV)
    python pipeline_DT.py --quick      # Quick mode (10 trials, 5×1 CV)
"""

import sys
import os

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_core import run_full_pipeline, parse_args


def main():
    """Run the DT prediction pipeline."""
    # Check if --quick flag is passed (allow running without temperature arg)
    quick = '--quick' in sys.argv

    print("=" * 70)
    print("  DEFORMATION TEMPERATURE (DT) PREDICTION")
    print("  Predicting initial deformation temperature of ash samples")
    print("=" * 70)

    results = run_full_pipeline(
        temperature='DT',
        data_dir='DATA',
        results_base='RESULTS',
        quick=quick,
    )

    # Print final summary
    if results:
        print("\n" + "=" * 70)
        print("  DT FINAL RESULTS")
        print("=" * 70)
        for ash_type, res in results.items():
            label = ash_type.replace(' ash', '').capitalize()
            print(f"  {label}: Best = {res['best_model']} "
                  f"(Test R² = {res['best_test_r2']:.4f})")
            print(f"    Results: {res['results_dir']}")
    else:
        print("\n  [ERROR] No results generated for DT")

    return results


if __name__ == '__main__':
    main()
