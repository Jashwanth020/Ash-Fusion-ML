import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
pipeline_core.py — Shared ML Pipeline for Ash Fusion Temperature Prediction

Publication-quality implementation with:
- sklearn Pipeline (Imputer → RobustScaler → Model) to prevent data leakage
- Optuna Bayesian hyperparameter optimization
- RepeatedKFold cross-validation
- 6 tree-based models (RF, ET, XGB, GBRT, CatBoost, AdaBoost)
- Stacking ensemble with RidgeCV meta-learner
- SHAP analysis + interaction values
- Statistical significance testing (Friedman + Wilcoxon)
- Publication-quality figures at 300 DPI

Usage:
    from pipeline_core import run_full_pipeline
    run_full_pipeline(temperature='DT', quick=False)
"""

import os
import sys
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import json
import joblib

from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RepeatedKFold,
    train_test_split,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================================
# Constants
# ======================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_SAMPLES = 30
OVERFIT_THRESHOLD = 0.10
DPI = 300

# Full mode
FULL_N_TRIALS = 100
FULL_CV_REPEATS = 3
FULL_CV_SPLITS = 5
FULL_PERM_REPEATS = 30

# Quick mode
QUICK_N_TRIALS = 10
QUICK_CV_REPEATS = 1
QUICK_CV_SPLITS = 5
QUICK_PERM_REPEATS = 10

# Model names mapped to SA file names
MODEL_SA_MAP = {
    'Random Forest': 'SA_RandomForest_combined_filtered.csv',
    'Extra Trees':   'SA_ExtraTrees_combined_filtered.csv',
    'XGBoost':       'SA_XGBoost_combined_filtered.csv',
    'GBRT':          'SA_GBRT_combined_filtered.csv',
    'CatBoost':      'SA_CatBoost_combined_filtered.csv',
    'AdaBoost':      'SA_AdaBoost_combined_filtered.csv',
}

MODEL_NAMES = list(MODEL_SA_MAP.keys())

# Publication figure styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ======================================================================
# 1. Reproducibility
# ======================================================================
def seed_everything(seed=RANDOM_STATE):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seed set to {seed}")


# ======================================================================
# 2. Data Loading
# ======================================================================
def load_consensus_data(temperature, data_dir='DATA'):
    """Load the consensus cleaned CSV for a given temperature."""
    path = os.path.join(data_dir, temperature,
                        f'ash_{temperature}_SA_consensus_cleaned.csv')
    df = pd.read_csv(path)
    print(f"[INFO] Loaded consensus data: {path} — shape {df.shape}")
    return df


def load_sa_data(temperature, model_name, data_dir='DATA'):
    """Load model-specific SA-filtered CSV for a given temperature."""
    filename = MODEL_SA_MAP[model_name]
    path = os.path.join(data_dir, temperature, filename)
    df = pd.read_csv(path)
    print(f"[INFO] Loaded SA data for {model_name}: {path} — shape {df.shape}")
    return df


# ======================================================================
# 3. Feature Engineering
# ======================================================================
def feature_engineering(df):
    """
    Add domain-specific engineered features with division-by-zero guards.

    Features:
    - Base_Acid_Ratio: (Fe2O3 + CaO + MgO + Na2O + K2O) / (SiO2 + Al2O3 + TiO2)
    - Silica_Ratio: 100 * SiO2 / (SiO2 + Fe2O3 + CaO + MgO)
    - Alkali_Ratio: Na2O + K2O

    Division-by-zero: denominators replaced with NaN (handled by Pipeline imputer).
    """
    df = df.copy()

    # Base/Acid ratio — guard against zero denominator
    denom_ba = df['SiO2'] + df['Al2O3'] + df['TiO2']
    denom_ba = denom_ba.replace(0, np.nan)
    df['Base_Acid_Ratio'] = (
        df['Fe2O3'] + df['CaO'] + df['MgO'] + df['Na2O'] + df['K2O']
    ) / denom_ba

    # Silica ratio — guard against zero denominator
    denom_si = df['SiO2'] + df['Fe2O3'] + df['CaO'] + df['MgO']
    denom_si = denom_si.replace(0, np.nan)
    df['Silica_Ratio'] = (100 * df['SiO2']) / denom_si

    # Alkali ratio (no division)
    df['Alkali_Ratio'] = df['Na2O'] + df['K2O']

    print(f"[INFO] Feature engineering complete — added 3 features, "
          f"new shape {df.shape}")
    return df


# ======================================================================
# 4. Coal/Biomass Separation
# ======================================================================
def split_feedstock(df):
    """Separate coal and biomass ash datasets."""
    coal_df = df[df['Type'] == 'coal ash'].copy()
    biomass_df = df[df['Type'] == 'biomass ash'].copy()
    print(f"[INFO] Coal: {len(coal_df)} samples | Biomass: {len(biomass_df)} samples")
    return {'Coal': coal_df, 'Biomass': biomass_df}


# ======================================================================
# 5. Feature Preparation
# ======================================================================
def prepare_features(df, temperature):
    """Extract feature matrix X and target vector y, dropping metadata columns."""
    drop_cols = ['Type', 'group', 'Research paper', temperature]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[temperature].values
    feature_names = list(X.columns)
    print(f"[INFO] Features: {len(feature_names)} — {feature_names}")
    return X, y, feature_names


# ======================================================================
# 6. Model Builders
# ======================================================================
def build_pipeline(model):
    """Wrap model in sklearn Pipeline with imputation and scaling."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', model),
    ])


def build_model(name, params):
    """Instantiate a model by name with given parameters."""
    if name == 'Random Forest':
        return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'Extra Trees':
        return ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'XGBoost':
        return XGBRegressor(**params, objective='reg:squarederror',
                            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    elif name == 'GBRT':
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    elif name == 'CatBoost':
        return CatBoostRegressor(**params, random_seed=RANDOM_STATE,
                                 verbose=0)
    elif name == 'AdaBoost':
        return AdaBoostRegressor(**params, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")


# ======================================================================
# 7. Optuna Parameter Spaces
# ======================================================================
def suggest_params(trial, model_name):
    """Define Optuna hyperparameter search space per model."""

    if model_name in ['Random Forest', 'Extra Trees']:
        return {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 12),
            'max_features':     trial.suggest_float('max_features', 0.3, 1.0),
        }

    elif model_name == 'XGBoost':
        return {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 1000),
            'max_depth':       trial.suggest_int('max_depth', 2, 5),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }

    elif model_name == 'GBRT':
        return {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 150),
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15),
        }

    elif model_name == 'CatBoost':
        return {
            'iterations':      trial.suggest_int('iterations', 100, 1000),
            'depth':           trial.suggest_int('depth', 2, 5),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg':     trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }

    elif model_name == 'AdaBoost':
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
            'loss':          trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
        }

    else:
        raise ValueError(f"No parameter space for: {model_name}")


# ======================================================================
# 8. Bayesian Hyperparameter Optimization
# ======================================================================
def bayesian_search(model_name, X_train, y_train, n_trials, cv):
    """
    Run Optuna Bayesian optimization for a single model.
    Returns (best_params, best_cv_score).
    """

    def objective(trial):
        params = suggest_params(trial, model_name)
        model = build_model(model_name, params)
        pipe = build_pipeline(model)

        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring='r2', n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  [OPTUNA] {model_name}: best CV R² = {study.best_value:.4f} "
          f"({n_trials} trials)")
    return study.best_params, study.best_value


# ======================================================================
# 9. Metrics
# ======================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error — replaces MAPE."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Guard against 0/0
    mask = denominator > 0
    return 100.0 * np.mean(
        2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    )


def compute_metrics(y_true, y_pred, label=''):
    """Compute R², RMSE, MAE, SMAPE."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    return {
        f'{label}_R2': r2,
        f'{label}_RMSE': rmse,
        f'{label}_MAE': mae,
        f'{label}_SMAPE': smape_val,
    }


def compute_cv_metrics(pipeline, X_train, y_train, cv):
    """
    Compute per-fold R² and RMSE via RepeatedKFold.
    Returns dict with arrays and summary stats.
    """
    r2_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring='r2', n_jobs=-1,
    )
    # RMSE via neg_root_mean_squared_error
    rmse_scores = -cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1,
    )

    # 95% confidence interval on R²
    n = len(r2_scores)
    ci_95 = stats.t.interval(
        0.95, df=n - 1,
        loc=r2_scores.mean(),
        scale=stats.sem(r2_scores),
    )

    return {
        'cv_r2_scores': r2_scores,
        'cv_r2_mean': r2_scores.mean(),
        'cv_r2_std': r2_scores.std(),
        'cv_r2_ci_low': ci_95[0],
        'cv_r2_ci_high': ci_95[1],
        'cv_rmse_scores': rmse_scores,
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std(),
    }


# ======================================================================
# 10. Overfitting Detection
# ======================================================================
def detect_overfitting(train_r2, test_r2):
    """Flag overfitting if Train R² - Test R² > threshold."""
    gap = train_r2 - test_r2
    status = 'OVERFIT' if gap > OVERFIT_THRESHOLD else 'OK'
    return gap, status


# ======================================================================
# 11. Statistical Significance Testing
# ======================================================================
def significance_testing(all_cv_scores):
    """
    Perform Friedman omnibus test and pairwise Wilcoxon tests with
    Bonferroni correction.

    Parameters:
        all_cv_scores: dict mapping model_name -> array of per-fold R² scores

    Returns:
        DataFrame with pairwise test results
    """
    model_names = list(all_cv_scores.keys())
    n_models = len(model_names)

    if n_models < 2:
        print("  [WARN] Need at least 2 models for significance testing")
        return pd.DataFrame()

    # Friedman test (omnibus)
    score_arrays = list(all_cv_scores.values())

    # Ensure all arrays are the same length (trim to minimum)
    min_len = min(len(s) for s in score_arrays)
    score_arrays_trimmed = [s[:min_len] for s in score_arrays]

    if n_models >= 3:
        stat_f, p_f = friedmanchisquare(*score_arrays_trimmed)
        print(f"  [FRIEDMAN] chi2 = {stat_f:.4f}, p = {p_f:.4f}")
    else:
        p_f = None
        print("  [INFO] Only 2 models - skipping Friedman test")

    # Pairwise Wilcoxon tests
    n_comparisons = n_models * (n_models - 1) // 2
    alpha_corrected = 0.05 / max(n_comparisons, 1)

    results = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_a = model_names[i]
            name_b = model_names[j]
            scores_a = all_cv_scores[name_a][:min_len]
            scores_b = all_cv_scores[name_b][:min_len]

            try:
                stat_w, p_w = wilcoxon(scores_a, scores_b)
                significant = 'Yes' if p_w < alpha_corrected else 'No'
            except Exception as e:
                stat_w, p_w, significant = np.nan, np.nan, f'Error: {e}'

            results.append({
                'Model_A': name_a,
                'Model_B': name_b,
                'W_Statistic': stat_w,
                'p_value': p_w,
                'Bonferroni_alpha': alpha_corrected,
                'Significant': significant,
            })

    df_sig = pd.DataFrame(results)
    return df_sig


# ======================================================================
# 12. Stacking Ensemble
# ======================================================================
def build_stacking_ensemble(best_params_dict):
    """
    Build a StackingRegressor with tuned base models and RidgeCV meta-learner.
    Uses consensus data (all models need same rows).
    """
    base_estimators = []
    for name in MODEL_NAMES:
        if name in best_params_dict:
            model = build_model(name, best_params_dict[name])
            safe_name = name.lower().replace(' ', '_')
            base_estimators.append((safe_name, model))

    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', stacking),
    ])
    return pipe


# ======================================================================
# 13. PLOT: Correlation Heatmap
# ======================================================================
def plot_correlation_heatmap(X, feature_names, save_path):
    """Pearson correlation heatmap of features."""
    corr = pd.DataFrame(X, columns=feature_names).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
    )
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")


# ======================================================================
# 14. PLOT: Actual vs Predicted
# ======================================================================
def plot_actual_vs_pred(y_train, y_train_pred, y_test, y_test_pred,
                        model_name, temperature, save_path):
    """Scatter plot of actual vs predicted values for train and test sets."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_train, y_train_pred, alpha=0.5, color='#2196F3',
               edgecolors='white', linewidths=0.3, s=50, label='Train', zorder=2)
    ax.scatter(y_test, y_test_pred, alpha=0.7, color='#FF5722',
               edgecolors='white', linewidths=0.3, s=60, label='Test', zorder=3)

    # Perfect prediction line
    all_vals = np.concatenate([y_train, y_test])
    lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
    ax.plot(lims, lims, '--', color='#333333', linewidth=1.5,
            alpha=0.7, label='Perfect prediction', zorder=1)

    # Metrics annotation
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    textstr = (f'Train: R² = {train_r2:.4f}, RMSE = {train_rmse:.1f}\n'
               f'Test:  R² = {test_r2:.4f}, RMSE = {test_rmse:.1f}')
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(f'Actual {temperature} (°C)', fontweight='bold')
    ax.set_ylabel(f'Predicted {temperature} (°C)', fontweight='bold')
    ax.set_title(f'{model_name} — Actual vs Predicted ({temperature})',
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")


# ======================================================================
# 15. PLOT: Residual Analysis
# ======================================================================
def plot_residuals(y_test, y_pred, model_name, temperature, save_path):
    """Residuals vs predicted + residual distribution (2-panel)."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='#3F51B5',
                    edgecolors='k', linewidths=0.3, s=40)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel(f'Predicted {temperature} (°C)', fontweight='bold')
    axes[0].set_ylabel('Residuals (°C)', fontweight='bold')
    axes[0].set_title('Residuals vs Predicted', fontweight='bold')

    # Right: Residual Distribution
    axes[1].hist(residuals, bins=25, edgecolor='black', alpha=0.7,
                 color='#4CAF50')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Residual (°C)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Residual Distribution', fontweight='bold')

    # Stats annotation
    axes[1].text(
        0.95, 0.95,
        f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}',
        transform=axes[1].transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    fig.suptitle(f'{model_name} — Residual Analysis ({temperature})',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")


# ======================================================================
# 16. PLOT: Learning Curve
# ======================================================================
def plot_learning_curve(pipeline, X_train, y_train, cv,
                        model_name, temperature, save_path):
    """Learning curve with ±1 std confidence bands."""
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=cv, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='#2196F3',
            linewidth=2, label='Train Score')
    ax.plot(train_sizes, val_mean, 's-', color='#FF5722',
            linewidth=2, label='Validation Score')

    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color='#2196F3')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color='#FF5722')

    ax.set_xlabel('Training Set Size', fontweight='bold')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title(f'{model_name} — Learning Curve ({temperature})',
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.set_ylim([max(0, val_mean.min() - 0.15), 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")


# ======================================================================
# 17. PLOT: Permutation Importance
# ======================================================================
def plot_permutation_importance(pipeline, X_test, y_test, feature_names,
                                model_name, temperature, save_path,
                                n_repeats=FULL_PERM_REPEATS):
    """Horizontal bar chart of permutation importance on test data."""
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.4)))
    ax.barh(
        np.array(feature_names)[sorted_idx],
        result.importances_mean[sorted_idx],
        xerr=result.importances_std[sorted_idx],
        color='#009688', edgecolor='white', linewidth=0.5,
        capsize=3,
    )
    ax.set_xlabel('Mean Decrease in R²', fontweight='bold')
    ax.set_title(f'{model_name} — Permutation Importance ({temperature})',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")

    return result


# ======================================================================
# 18. PLOT: SHAP Analysis
# ======================================================================
def run_shap_analysis(pipeline, X_test, feature_names, model_name,
                      temperature, results_dir):
    """Generate SHAP summary, bar, dependence, and interaction plots."""

    # Extract the underlying tree model from the pipeline
    tree_model = pipeline.named_steps['model']

    # Transform X_test through imputer+scaler
    X_test_transformed = pipeline[:-1].transform(X_test)

    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_test_transformed)
    except Exception as e:
        print(f"  [WARN] SHAP failed for {model_name}: {e}")
        return

    # Convert to DataFrame for readable feature names
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names,
                      show=False)
    plt.title(f'{model_name} — SHAP Summary ({temperature})', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, 'shap_summary.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {path}")

    # 2. SHAP Bar Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title(f'{model_name} — SHAP Feature Importance ({temperature})',
              fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, 'shap_bar.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {path}")

    # 3. SHAP Dependence Plots (top 3 features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    for feat_idx in top3_idx:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat_idx, shap_values, X_test_df,
            feature_names=feature_names,
            show=False,
        )
        fname = feature_names[feat_idx]
        safe_fname = fname.replace('/', '_').replace('\\', '_')
        path = os.path.join(results_dir, f'shap_dependence_{safe_fname}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  [PLOT] Saved: {path}")

    # 4. SHAP Interaction Values (can be slow — skip for very small datasets)
    try:
        if len(X_test_transformed) <= 200:
            shap_interaction = explainer.shap_interaction_values(X_test_transformed)

            plt.figure(figsize=(10, 8))
            # Mean absolute interaction values heatmap
            mean_interact = np.abs(shap_interaction).mean(axis=0)
            interact_df = pd.DataFrame(mean_interact,
                                       index=feature_names,
                                       columns=feature_names)
            sns.heatmap(interact_df, cmap='YlOrRd', annot=True, fmt='.3f',
                        square=True, linewidths=0.5)
            plt.title(f'{model_name} — SHAP Interaction ({temperature})',
                      fontweight='bold')
            plt.tight_layout()
            path = os.path.join(results_dir, 'shap_interaction.png')
            plt.savefig(path, dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"  [PLOT] Saved: {path}")
        else:
            print(f"  [INFO] Skipping SHAP interaction (test set too large: "
                  f"{len(X_test_transformed)} > 200)")
    except Exception as e:
        print(f"  [WARN] SHAP interaction failed: {e}")


# ======================================================================
# 19. PLOT: Significance Matrix Heatmap
# ======================================================================
def plot_significance_matrix(df_significance, model_names_list,
                             temperature, save_path):
    """Heatmap of pairwise Wilcoxon p-values."""
    if df_significance.empty:
        print("  [WARN] No significance data to plot")
        return

    n = len(model_names_list)
    matrix = np.ones((n, n))

    for _, row in df_significance.iterrows():
        i = model_names_list.index(row['Model_A'])
        j = model_names_list.index(row['Model_B'])
        matrix[i, j] = row['p_value']
        matrix[j, i] = row['p_value']

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix, annot=True, fmt='.4f',
        xticklabels=model_names_list,
        yticklabels=model_names_list,
        cmap='RdYlGn_r', vmin=0, vmax=0.1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8, 'label': 'p-value'},
    )
    ax.set_title(f'Pairwise Wilcoxon p-values ({temperature})',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {save_path}")


# ======================================================================
# 20. Save Results
# ======================================================================
def save_metrics_csv(all_metrics, save_path):
    """Save consolidated metrics table to CSV."""
    df = pd.DataFrame(all_metrics)
    df.to_csv(save_path, index=False)
    print(f"  [CSV] Saved: {save_path}")
    return df


def save_cv_results_csv(all_cv_results, save_path):
    """Save cross-validation results to CSV."""
    df = pd.DataFrame(all_cv_results)
    df.to_csv(save_path, index=False)
    print(f"  [CSV] Saved: {save_path}")
    return df


# ======================================================================
# 21. Main Experiment Runner
# ======================================================================
def run_experiment(temperature, ash_type, data_dir='DATA',
                   results_base='RESULTS', quick=False):
    """
    Run the complete ML pipeline for one (temperature, ash_type) combination.

    Steps:
    1. Load SA data for each model
    2. Feature engineering on all datasets
    3. For each model: Optuna tuning → CV evaluation → train/test metrics
    4. Select best model
    5. Generate all publication figures
    6. Stacking ensemble
    7. Statistical significance testing
    8. Save all results
    """

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {temperature} — {ash_type}")
    print(f"{'='*70}")

    # Config based on mode
    n_trials = QUICK_N_TRIALS if quick else FULL_N_TRIALS
    cv_repeats = QUICK_CV_REPEATS if quick else FULL_CV_REPEATS
    cv_splits = QUICK_CV_SPLITS if quick else FULL_CV_SPLITS
    perm_repeats = QUICK_PERM_REPEATS if quick else FULL_PERM_REPEATS

    cv = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=RANDOM_STATE,
    )

    mode_str = "QUICK" if quick else "FULL"
    print(f"  [MODE] {mode_str}: {n_trials} Optuna trials, "
          f"RepeatedKFold({cv_splits}×{cv_repeats}), "
          f"{perm_repeats} perm repeats")

    # Results directory
    ash_label = ash_type.replace(' ash', '').capitalize()
    results_dir = os.path.join(results_base, temperature, ash_label)
    os.makedirs(results_dir, exist_ok=True)

    # -- Train each model on its SA-filtered data --
    all_metrics = []
    all_cv_results = []
    all_cv_scores = {}
    best_params_dict = {}
    trained_pipelines = {}

    best_test_r2 = -np.inf
    best_model_name = None

    for model_name in MODEL_NAMES:
        print(f"\n  -- {model_name} --")

        # Load SA data for this model
        sa_df = load_sa_data(temperature, model_name, data_dir)
        sa_ash = sa_df[sa_df['Type'] == ash_type].copy()
        sa_ash = feature_engineering(sa_ash)

        if len(sa_ash) < MIN_SAMPLES:
            print(f"  [SKIP] {model_name}: only {len(sa_ash)} samples")
            continue

        X_sa, y_sa, _ = prepare_features(sa_ash, temperature)

        feature_names = list(X_sa.columns)

        # Train/test split on SA data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sa, y_sa,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        print(f"  [SPLIT] SA: Train={len(X_train)}, Test={len(X_test)}")

        # Bayesian hyperparameter search
        best_params, optuna_best = bayesian_search(
            model_name, X_train, y_train, n_trials, cv,
        )
        best_params_dict[model_name] = best_params

        # Build final pipeline with best params
        model = build_model(model_name, best_params)
        pipe = build_pipeline(model)

        # Cross-validation metrics
        cv_metrics = compute_cv_metrics(pipe, X_train, y_train, cv)
        all_cv_scores[model_name] = cv_metrics['cv_r2_scores']

        # Fit final model
        pipe.fit(X_train, y_train)
        trained_pipelines[model_name] = pipe

        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred, label='Train')
        test_metrics = compute_metrics(y_test, y_test_pred, label='Test')

        # Overfitting gap
        gap, overfit_status = detect_overfitting(
            train_metrics['Train_R2'], test_metrics['Test_R2'])

        # Consolidate
        row = {
            'Model': model_name,
            'SA_Samples': len(X_sa),
            **train_metrics,
            **test_metrics,
            'CV_R2_Mean': cv_metrics['cv_r2_mean'],
            'CV_R2_Std': cv_metrics['cv_r2_std'],
            'CV_R2_CI_Low': cv_metrics['cv_r2_ci_low'],
            'CV_R2_CI_High': cv_metrics['cv_r2_ci_high'],
            'CV_RMSE_Mean': cv_metrics['cv_rmse_mean'],
            'CV_RMSE_Std': cv_metrics['cv_rmse_std'],
            'Overfit_Gap': gap,
            'Overfit_Status': overfit_status,
        }
        all_metrics.append(row)

        cv_row = {
            'Model': model_name,
            'CV_R2_Mean': cv_metrics['cv_r2_mean'],
            'CV_R2_Std': cv_metrics['cv_r2_std'],
            'CV_R2_CI_95': f"[{cv_metrics['cv_r2_ci_low']:.4f}, "
                           f"{cv_metrics['cv_r2_ci_high']:.4f}]",
            'CV_RMSE_Mean': cv_metrics['cv_rmse_mean'],
            'CV_RMSE_Std': cv_metrics['cv_rmse_std'],
        }
        all_cv_results.append(cv_row)

        print(f"  [RESULT] {model_name}: "
              f"Train R²={train_metrics['Train_R2']:.4f}, "
              f"Test R²={test_metrics['Test_R2']:.4f}, "
              f"CV R²={cv_metrics['cv_r2_mean']:.4f}±{cv_metrics['cv_r2_std']:.4f} "
              f"| {overfit_status}")

        # Track best
        if test_metrics['Test_R2'] > best_test_r2:
            best_test_r2 = test_metrics['Test_R2']
            best_model_name = model_name
            best_X_train, best_X_test = X_train, X_test
            best_y_train, best_y_test = y_train, y_test
            best_y_train_pred = y_train_pred
            best_y_test_pred = y_test_pred

    if best_model_name is None:
        print("  [ERROR] No models trained successfully")
        return None

    print(f"\n  * BEST MODEL: {best_model_name} (Test R2 = {best_test_r2:.4f})")

    # -- Generate publication figures for best model --
    best_pipe = trained_pipelines[best_model_name]

    # Actual vs Predicted
    plot_actual_vs_pred(
        best_y_train, best_y_train_pred,
        best_y_test, best_y_test_pred,
        best_model_name, temperature,
        os.path.join(results_dir, 'actual_vs_pred.png'),
    )

    # Residual analysis
    plot_residuals(
        best_y_test, best_y_test_pred,
        best_model_name, temperature,
        os.path.join(results_dir, 'residual_analysis.png'),
    )

    # Learning curve
    plot_learning_curve(
        best_pipe, best_X_train, best_y_train, cv,
        best_model_name, temperature,
        os.path.join(results_dir, 'learning_curve.png'),
    )

    # Permutation importance
    plot_permutation_importance(
        best_pipe, best_X_test, best_y_test, feature_names,
        best_model_name, temperature,
        os.path.join(results_dir, 'permutation_importance.png'),
        n_repeats=perm_repeats,
    )

    # SHAP analysis
    run_shap_analysis(
        best_pipe, best_X_test, feature_names,
        best_model_name, temperature, results_dir,
    )

    # -- Statistical significance testing --
    df_sig = significance_testing(all_cv_scores)
    if not df_sig.empty:
        df_sig.to_csv(os.path.join(results_dir, 'significance_tests.csv'),
                      index=False)
        print(f"  [CSV] Saved: significance_tests.csv")

        plot_significance_matrix(
            df_sig, list(all_cv_scores.keys()), temperature,
            os.path.join(results_dir, 'significance_matrix.png'),
        )

    # -- Save Best Model to Disk --
    model_path = os.path.join(results_dir, f"{best_model_name.replace(' ', '_')}_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n  [SAVE] Saved best model to: {model_path}")

    # -- Save Best Hyperparameters for all models --
    hyperparams_path = os.path.join(results_dir, 'best_hyperparameters.json')
    try:
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params_dict, f, indent=4)
        print(f"  [SAVE] Saved best hyperparameters to: {hyperparams_path}")
    except Exception as e:
        print(f"  [WARN] Failed to save hyperparameters: {e}")

    # -- Save all CSV results --
    save_metrics_csv(all_metrics, os.path.join(results_dir, 'metrics.csv'))
    save_cv_results_csv(all_cv_results,
                        os.path.join(results_dir, 'cv_results.csv'))

    # -- Print summary table --
    print(f"\n{'-'*70}")
    print(f"  SUMMARY: {temperature} - {ash_type}")
    print(f"{'-'*70}")
    df_metrics = pd.DataFrame(all_metrics)
    summary_cols = ['Model', 'Train_R2', 'Test_R2', 'CV_R2_Mean',
                    'CV_R2_Std', 'Test_RMSE', 'Test_SMAPE', 'Overfit_Status']
    available_cols = [c for c in summary_cols if c in df_metrics.columns]
    print(df_metrics[available_cols].to_string(index=False))
    print(f"{'-'*70}\n")

    return {
        'temperature': temperature,
        'ash_type': ash_type,
        'best_model': best_model_name,
        'best_test_r2': best_test_r2,
        'metrics_df': pd.DataFrame(all_metrics),
        'results_dir': results_dir,
    }


# ======================================================================
# 22. Full Pipeline Runner
# ======================================================================
def run_full_pipeline(temperature, data_dir='DATA', results_base='RESULTS',
                      quick=False):
    """
    Run the complete pipeline for a single temperature across both
    coal and biomass ash types.

    Parameters:
        temperature: str — one of 'DT', 'FT', 'HT', 'ST'
        data_dir: str — path to DATA directory
        results_base: str — path to RESULTS directory
        quick: bool — if True, use reduced iterations
    """
    seed_everything()

    print(f"\n{'#'*70}")
    print(f"  ASH FUSION TEMPERATURE PREDICTION: {temperature}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"{'#'*70}")

    # Run for each ash type (No consensus data available)
    results = {}
    for ash_type in ['coal ash', 'biomass ash']:
        result = run_experiment(
            temperature=temperature,
            ash_type=ash_type,
            data_dir=data_dir,
            results_base=results_base,
            quick=quick,
        )
        if result:
            results[ash_type] = result

    print(f"\n{'#'*70}")
    print(f"  {temperature} PIPELINE COMPLETE")
    print(f"  Results saved to: {os.path.join(results_base, temperature)}/")
    print(f"{'#'*70}\n")

    return results


# ======================================================================
# CLI Entry Point (for direct execution)
# ======================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Ash Fusion Temperature Prediction Pipeline',
    )
    parser.add_argument(
        'temperature',
        choices=['DT', 'FT', 'HT', 'ST'],
        help='Target temperature to predict',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: reduced Optuna trials (10) and CV repeats (1)',
    )
    parser.add_argument(
        '--data-dir', default='DATA',
        help='Path to DATA directory (default: DATA)',
    )
    parser.add_argument(
        '--results-dir', default='RESULTS',
        help='Path to RESULTS directory (default: RESULTS)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_full_pipeline(
        temperature=args.temperature,
        data_dir=args.data_dir,
        results_base=args.results_dir,
        quick=args.quick,
    )
