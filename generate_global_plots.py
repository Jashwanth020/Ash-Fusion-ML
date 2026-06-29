"""
generate_global_plots.py — Generate SHAP, Feature Importance, and Scatter plots 
for the full dataset using the best models.
Displays Test R² in the Scatter plot titles.
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
RESULTS_DIR = os.path.join('RESULTS', 'GLOBAL_PLOTS')
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    print(f"Generating Global Plots in {RESULTS_DIR}...")
    
    for (temp, feedstock), (model_name, filename) in BEST_MODELS.items():
        print(f"\nProcessing {temp} {feedstock} ({model_name})...")
        path = os.path.join(DATA_DIR, temp, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue
            
        df = pd.read_csv(path)
        ash_type = 'coal ash' if feedstock == 'Coal' else 'biomass ash'
        df = df[df['Type'] == ash_type].copy()
        
        if len(df) < 30:
            print(f"  [SKIP] Too few samples ({len(df)})")
            continue
            
        df = feature_engineering(df)
        df.dropna(subset=[temp], inplace=True)
        
        drop_cols = ['Type', 'group', 'Research paper', temp]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[temp].values
        feature_names = list(X.columns)
        
        # Split into train/test to get the true Test R2 score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Preprocessor
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Fit preprocessor on train and transform all
        preprocessor.fit(X_train)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        X_all_trans = preprocessor.transform(X)
        
        # Train model strictly on training data
        model = get_model(model_name)
        model.fit(X_train_trans, y_train)
        
        # Predict on Test data for the Test R2 score
        y_test_pred = model.predict(X_test_trans)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Predict on ALL data for the scatter plot
        y_all_pred = model.predict(X_all_trans)
        
        # 1. Scatter Plot (Actual vs Predicted on ALL data, but showing TEST R2)
        plt.figure(figsize=(6, 5))
        plt.scatter(y, y_all_pred, alpha=0.6, edgecolors='k', color='#2b5c8f' if feedstock == 'Coal' else '#d95f02')
        min_val = min(y.min(), y_all_pred.min())
        max_val = max(y.max(), y_all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel(f'Actual {temp} (°C)', fontweight='bold')
        plt.ylabel(f'Predicted {temp} (°C)', fontweight='bold')
        plt.title(f'Actual vs Predicted ({temp} {feedstock})\nTest R²={test_r2:.3f}, Test RMSE={test_rmse:.1f}°C', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'scatter_{temp}_{feedstock}.png'))
        plt.close()
        
        # 2. Feature Importance Plot (based on the model trained on 80%)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_names))
        
        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), importances[indices][:top_n][::-1], color='#2b5c8f' if feedstock == 'Coal' else '#d95f02')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n][::-1])
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances ({temp} {feedstock} - {model_name})', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'fi_{temp}_{feedstock}.png'))
        plt.close()
        
        # 3. SHAP Summary Plot (on ALL data for global overview)
        plt.figure()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_all_trans)
        
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
            
        shap.summary_plot(shap_values, X_all_trans, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot ({temp} {feedstock} - {model_name})', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_{temp}_{feedstock}.png'))
        plt.close()
        
        print(f"  [SAVED] {temp} {feedstock} plots.")

if __name__ == '__main__':
    main()
