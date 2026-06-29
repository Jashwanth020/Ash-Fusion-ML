"""
pipeline_FT.py — Flow Temperature (FT) Prediction Pipeline

Predicts FT (flow temperature) of ash from coal and biomass feedstocks
using 6 tree-based ML models with Bayesian hyperparameter optimization.

Usage:
    python pipeline_FT.py              # Full mode (50 Optuna trials, 5×3 CV)
    python pipeline_FT.py --quick      # Quick mode (10 trials, 5×1 CV)
"""

import sys
import os

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_core import run_full_pipeline, parse_args


def main():
    """Run the FT prediction pipeline."""
    quick = '--quick' in sys.argv

    print("=" * 70)
    print("  FLOW TEMPERATURE (FT) PREDICTION")
    print("  Predicting flow temperature of ash samples")
    print("=" * 70)

    results = run_full_pipeline(
        temperature='FT',
        data_dir='DATA',
        results_base='RESULTS',
        quick=quick,
    )

    # Print final summary
    if results:
        print("\n" + "=" * 70)
        print("  FT FINAL RESULTS")
        print("=" * 70)
        for ash_type, res in results.items():
            label = ash_type.replace(' ash', '').capitalize()
            print(f"  {label}: Best = {res['best_model']} "
                  f"(Test R² = {res['best_test_r2']:.4f})")
            print(f"    Results: {res['results_dir']}")
    else:
        print("\n  [ERROR] No results generated for FT")

    return results


if __name__ == '__main__':
    main()
