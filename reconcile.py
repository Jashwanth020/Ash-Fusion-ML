"""
reconcile.py
============
Post-prediction monotonic constraint enforcement for ash fusion temperatures.
Enforces: DT <= ST <= HT <= FT
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

# ── Column name config ────────────────────────────────────────────────────────

PRED_COLS = ['DT', 'ST', 'HT', 'FT']          # raw prediction columns
PAV_COLS  = ['DT_pav', 'ST_pav', 'HT_pav', 'FT_pav']
CLSO_COLS = ['DT_clso', 'ST_clso', 'HT_clso', 'FT_clso']

# ── Violation diagnosis ───────────────────────────────────────────────────────

def diagnose_violations(df, pred_cols=PRED_COLS, verbose=True):
    """
    Reports violation rate and magnitude for each consecutive temperature pair.
    Returns a boolean Series: True where at least one pair is violated.
    """
    pairs = [
        (pred_cols[0], pred_cols[1], 'DT > ST'),
        (pred_cols[1], pred_cols[2], 'ST > HT'),
        (pred_cols[2], pred_cols[3], 'HT > FT'),
    ]

    any_violation = np.zeros(len(df), dtype=bool)

    if verbose:
        print("\n" + "=" * 50)
        print("  VIOLATION DIAGNOSIS")
        print("=" * 50)

    for lower, upper, label in pairs:
        mask = df[lower] > df[upper]
        mag  = (df[lower] - df[upper]).clip(lower=0)
        any_violation |= mask

        if verbose:
            print(f"\n  {label}")
            print(f"    Violated : {mask.sum()}/{len(df)} ({100*mask.mean():.1f}%)")
            if mask.any():
                print(f"    Mean mag : {mag[mask].mean():.2f} °C")
                print(f"    Max  mag : {mag.max():.2f} °C")

    if verbose:
        print(f"\n  ANY violation : {any_violation.sum()}/{len(df)} "
              f"({100*any_violation.mean():.1f}%)")
        print("=" * 50)

    return pd.Series(any_violation, index=df.index, name='violated')

# ── Method 1: Isotonic Regression (PAV) ──────────────────────────────────────

def _pav_row(row, pred_cols, out_cols):
    ir = IsotonicRegression(increasing=True)
    values = row[pred_cols].values.astype(float)
    corrected = ir.fit_transform(np.arange(4), values)
    return pd.Series(corrected, index=out_cols)

def apply_pav(df, pred_cols=PRED_COLS, out_cols=PAV_COLS):
    """
    Applies Pool Adjacent Violators algorithm row-wise.
    Finds nearest non-decreasing sequence in least-squares sense.
    Guaranteed to produce 0% violation rate.
    """
    return df.apply(_pav_row, axis=1, pred_cols=pred_cols, out_cols=out_cols)

# ── Method 2: Constrained Least-Squares Optimization (CLSO) ─────────────────

def _clso_row(row, pred_cols, out_cols, weights=None, epsilon=10.0):
    y = row[pred_cols].values.astype(float)
    w = np.ones(4) if weights is None else np.array(weights)

    def objective(x):
        return np.sum(w * (x - y) ** 2)

    def grad(x):
        return 2 * w * (x - y)

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - epsilon},   # ST >= DT + epsilon
        {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - epsilon},   # HT >= ST + epsilon
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - epsilon},   # FT >= HT + epsilon
    ]

    result = minimize(
        objective, y, jac=grad,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    return pd.Series(result.x, index=out_cols)

def apply_clso(df, pred_cols=PRED_COLS, out_cols=CLSO_COLS, weights=None, epsilon=10.0):
    """
    Solves a constrained quadratic program per row.
    Equivalent to PAV when weights=None and epsilon=0.
    Supports non-uniform weights and strict separation margins.
    """
    return df.apply(
        _clso_row, axis=1,
        pred_cols=pred_cols, out_cols=out_cols, weights=weights, epsilon=epsilon
    )

# ── Comparison summary ────────────────────────────────────────────────────────

def _violation_rate(df, cols):
    mask = np.zeros(len(df), dtype=bool)
    for a, b in zip(cols, cols[1:]):
        mask |= (df[a] > df[b])
    return 100 * mask.mean()

def _mean_adj(df, src_cols, tgt_cols):
    return np.mean([
        (df[t] - df[s]).abs().mean()
        for s, t in zip(src_cols, tgt_cols)
    ])

def compare_methods(df, pred_cols=PRED_COLS, obs_cols=None):
    """
    Prints a comparison table of violation rate and mean adjustment
    for raw predictions, PAV, and CLSO.
    obs_cols: list of observed ground-truth column names if available.
    """
    from sklearn.metrics import mean_absolute_error
    import math

    print("\n" + "=" * 65)
    print("  RECONCILIATION COMPARISON SUMMARY")
    print("=" * 65)

    methods = {
        'Raw predictions' : pred_cols,
        'PAV (isotonic)'  : PAV_COLS,
        'CLSO (opt.)'     : CLSO_COLS,
    }

    header = f"  {'Method':<22} {'Viol%':>7}  {'Mean Adj (°C)':>14}"
    if obs_cols:
        header += f"  {'MAE (°C)':>10}"
    print(header)
    print("  " + "-" * 60)

    for name, cols in methods.items():
        vr  = _violation_rate(df, cols)
        adj = _mean_adj(df, pred_cols, cols) if cols != pred_cols else 0.0
        row = f"  {name:<22} {vr:>6.1f}%  {adj:>14.3f}"

        if obs_cols:
            mae = np.mean([
                mean_absolute_error(df[o], df[c])
                for o, c in zip(obs_cols, cols)
            ])
            row += f"  {mae:>10.2f}"
        print(row)
    print("=" * 65)

# ── Main entry point ──────────────────────────────────────────────────────────

def reconcile(df, pred_cols=PRED_COLS, obs_cols=None, verbose=True, weights=None, epsilon=10.0):
    """
    Full reconciliation pipeline.
    """
    if verbose:
        diagnose_violations(df, pred_cols=pred_cols)
    
    pav_df  = apply_pav(df, pred_cols=pred_cols)
    clso_df = apply_clso(df, pred_cols=pred_cols, weights=weights, epsilon=epsilon)
    
    df = pd.concat([df, pav_df, clso_df], axis=1)
    
    if verbose:
        compare_methods(df, pred_cols=pred_cols, obs_cols=obs_cols)
    return df
