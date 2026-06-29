"""
generate_pca_pdp.py — Generate PCA and Partial Dependence Plots (PDP) for Ash Fusion

Creates:
1. PCA plots showing Coal vs Biomass ash composition clustering in 2D space.
2. PDP plots showing the marginal effects of the top 3 features for each best model.
Results saved in RESULTS/PCA_PDP/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RANDOM_STATE = 42
DATA_DIR = 'DATA'
RESULTS_DIR = os.path.join('RESULTS', 'PCA_PDP')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Best models mapping (from the 100-trial runs)
BEST_MODELS = {
    ('DT', 'Coal'): ('Extra Trees', 'SA_ExtraTrees_combined_filtered.csv'),
    ('DT', 'Biomass'): ('Extra Trees', 'SA_ExtraTrees_combined_filtered.csv'),
    ('FT', 'Coal'): ('Extra Trees', 'SA_ExtraTrees_combined_filtered.csv'),
    ('FT', 'Biomass'): ('Random Forest', 'SA_RandomForest_combined_filtered.csv'),
    ('HT', 'Coal'): ('XGBoost', 'SA_XGBoost_combined_filtered.csv'),
    ('HT', 'Biomass'): ('Extra Trees', 'SA_ExtraTrees_combined_filtered.csv'),
    ('ST', 'Coal'): ('XGBoost', 'SA_XGBoost_combined_filtered.csv'),
    ('ST', 'Biomass'): ('Random Forest', 'SA_RandomForest_combined_filtered.csv'),
}

def feature_engineering(df):
    """Adds the 3 baseline engineered features with division-by-zero guards."""
    df = df.copy()
    
    # Base/Acid ratio
    denom_ba = df['SiO2'] + df['Al2O3'] + df['TiO2']
    denom_ba = denom_ba.replace(0, np.nan)
    df['Base_Acid_Ratio'] = (df['Fe2O3'] + df['CaO'] + df['MgO'] + df['Na2O'] + df['K2O']) / denom_ba
    
    # Silica ratio
    denom_si = df['SiO2'] + df['Fe2O3'] + df['CaO'] + df['MgO']
    denom_si = denom_si.replace(0, np.nan)
    df['Silica_Ratio'] = (100 * df['SiO2']) / denom_si
    
    # Alkali ratio
    df['Alkali_Ratio'] = df['Na2O'] + df['K2O']
    
    return df

def get_model(name):
    """Helper to return a model instance with high-quality hyperparameters."""
    if name == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=8, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'Extra Trees':
        return ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'XGBoost':
        return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}")

def run_pca():
    """Generates PCA plots showing Coal vs Biomass composition clustering."""
    print("Generating PCA plots...")
    for temp in ['DT', 'FT', 'HT', 'ST']:
        path = os.path.join(DATA_DIR, temp, f'ash_{temp}_SA_consensus_cleaned.csv')
        if not os.path.exists(path):
            print(f"  [SKIP] Consensus file not found for {temp}")
            continue
            
        df = pd.read_csv(path)
        df.dropna(subset=[temp, 'Type'], inplace=True)
        df = feature_engineering(df)
        
        # Select composition features for PCA
        features = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'P2O5', 'K2O', 'TiO2', 'SO3', 
                    'Base_Acid_Ratio', 'Silica_Ratio', 'Alkali_Ratio']
        
        X = df[features]
        # Impute missing values for PCA
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Run PCA
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        plt.figure(figsize=(8, 6))
        coal_mask = df['Type'] == 'coal ash'
        biomass_mask = df['Type'] == 'biomass ash'
        
        plt.scatter(X_pca[coal_mask, 0], X_pca[coal_mask, 1], color='#2b5c8f', alpha=0.6, label='Coal Ash', s=40, edgecolors='none')
        plt.scatter(X_pca[biomass_mask, 0], X_pca[biomass_mask, 1], color='#d95f02', alpha=0.7, label='Biomass Ash', s=45, edgecolors='k', linewidths=0.5)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontweight='bold')
        plt.title(f'PCA of Ash Composition ({temp} Dataset)', fontweight='bold', pad=15)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.tight_layout()
        
        save_path = os.path.join(RESULTS_DIR, f'pca_{temp}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  [SAVED] PCA plot for {temp} to {save_path}")

def run_pdp():
    """Generates Partial Dependence Plots (PDP) for best models."""
    print("Generating PDP plots...")
    for (temp, feedstock), (model_name, filename) in BEST_MODELS.items():
        path = os.path.join(DATA_DIR, temp, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] SA file not found: {path}")
            continue
            
        df = pd.read_csv(path)
        ash_type = 'coal ash' if feedstock == 'Coal' else 'biomass ash'
        df = df[df['Type'] == ash_type].copy()
        
        if len(df) < 30:
            print(f"  [SKIP] {temp} {feedstock}: too few samples ({len(df)})")
            continue
            
        df = feature_engineering(df)
        df.dropna(subset=[temp], inplace=True)
        
        drop_cols = ['Type', 'group', 'Research paper', temp]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[temp].values
        feature_names = list(X.columns)
        
        # Build pipeline
        model = get_model(model_name)
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        pipe.fit(X, y)
        
        # Find top 3 features using permutation importance
        perm = permutation_importance(pipe, X, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        top3_indices = np.argsort(perm.importances_mean)[-3:][::-1]
        top3_features = [feature_names[idx] for idx in top3_indices]
        
        print(f"  {temp} {feedstock} Best Model: {model_name} | Top Features: {top3_features}")
        
        # Plot PDP
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        display = PartialDependenceDisplay.from_estimator(
            pipe, X, top3_features, ax=ax, 
            line_kw={"color": "#2b5c8f", "linewidth": 2}
        )
        
        fig.suptitle(f'Partial Dependence Plots — {temp} {feedstock} ({model_name})', fontweight='bold', fontsize=13, y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(RESULTS_DIR, f'pdp_{temp}_{feedstock}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] PDP plot for {temp} {feedstock} to {save_path}")

def main():
    print("=" * 80)
    print("  GENERATING PCA AND PARTIAL DEPENDENCE PLOTS (PDP)")
    print("=" * 80)
    run_pca()
    print("-" * 80)
    run_pdp()
    print("=" * 80)
    print("  ALL PLOTS COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
