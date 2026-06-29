"""
generate_test_plots.py — Generate SHAP, Feature Importance, and Scatter plots 
specifically on the TEST dataset for ST Coal and HT Coal.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
RESULTS_DIR = os.path.join('RESULTS', 'TEST_PLOTS')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGETS = {
    ('HT', 'Coal'): ('XGBoost', 'SA_XGBoost_combined_filtered.csv'),
    ('ST', 'Coal'): ('XGBoost', 'SA_XGBoost_combined_filtered.csv'),
}

def feature_engineering(df):
    df = df.copy()
    denom_ba = df['SiO2'] + df['Al2O3'] + df['TiO2']
    denom_ba = denom_ba.replace(0, np.nan)
    df['Base_Acid_Ratio'] = (df['Fe2O3'] + df['CaO'] + df['MgO'] + df['Na2O'] + df['K2O']) / denom_ba
    
    denom_si = df['SiO2'] + df['Fe2O3'] + df['CaO'] + df['MgO']
    denom_si = denom_si.replace(0, np.nan)
    df['Silica_Ratio'] = (100 * df['SiO2']) / denom_si
    
    df['Alkali_Ratio'] = df['Na2O'] + df['K2O']
    return df

def get_model(name):
    if name == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=8, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'Extra Trees':
        return ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'XGBoost':
        return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)

def main():
    print(f"Generating Test Plots in {RESULTS_DIR}...")
    
    for (temp, feedstock), (model_name, filename) in TARGETS.items():
        print(f"\nProcessing {temp} {feedstock} ({model_name})...")
        path = os.path.join(DATA_DIR, temp, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue
            
        df = pd.read_csv(path)
        ash_type = 'coal ash' if feedstock == 'Coal' else 'biomass ash'
        df = df[df['Type'] == ash_type].copy()
        
        df = feature_engineering(df)
        df.dropna(subset=[temp], inplace=True)
        
        drop_cols = ['Type', 'group', 'Research paper', temp]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[temp].values
        feature_names = list(X.columns)
        
        # Train / Test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Preprocessing
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        # Model
        model = get_model(model_name)
        model.fit(X_train_trans, y_train)
        
        y_pred = model.predict(X_test_trans)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 1. Scatter Plot (Actual vs Predicted on TEST SET)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', color='#2b5c8f' if feedstock == 'Coal' else '#d95f02')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel(f'Actual {temp} (°C)', fontweight='bold')
        plt.ylabel(f'Predicted {temp} (°C)', fontweight='bold')
        plt.title(f'Test Data Actual vs Predicted ({temp} {feedstock})\nTest R²={r2:.3f}, RMSE={rmse:.1f}°C', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'test_scatter_{temp}_{feedstock}.png'))
        plt.close()
        
        # 2. Feature Importance Plot
        # For tree-based models, feature_importances_ are trained on the training set,
        # but to see impact purely on the test set, we can either use permutation importance on the test set,
        # or stick to SHAP values computed on the test set. I'll compute permutation importance on TEST SET.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test_trans, y_test, n_repeats=10, random_state=RANDOM_STATE)
        importances = perm.importances_mean
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_names))
        
        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), importances[indices][:top_n][::-1], color='#2b5c8f' if feedstock == 'Coal' else '#d95f02')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n][::-1])
        plt.xlabel('Permutation Feature Importance (Test Set)', fontweight='bold')
        plt.title(f'Top {top_n} Test Feature Importances ({temp} {feedstock})', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'test_fi_{temp}_{feedstock}.png'))
        plt.close()
        
        # 3. SHAP Summary Plot (on TEST SET)
        plt.figure()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_trans)
        
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
            
        shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
        plt.title(f'Test Set SHAP Summary Plot ({temp} {feedstock})', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'test_shap_{temp}_{feedstock}.png'))
        plt.close()
        
        print(f"  [SAVED] {temp} {feedstock} test plots.")

if __name__ == '__main__':
    main()
