import os
import pandas as pd
import numpy as np

DATA_DIR = r'p:\antigravity\DATA'
NEW_DATA_PATH = r'p:\antigravity\new_data_predict\SA_ExtraTrees_all_types_removed.csv'

BEST_MODELS = {
    'Coal': ('DT', 'SA_ExtraTrees_combined_filtered.csv'),
    'Biomass': ('DT', 'SA_ExtraTrees_combined_filtered.csv'),
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

new_df = pd.read_csv(NEW_DATA_PATH)
new_df = feature_engineering(new_df)

for feedstock, (temp, filename) in BEST_MODELS.items():
    ash_type = 'coal ash' if feedstock == 'Coal' else 'biomass ash'
    
    # Original Data
    orig_path = os.path.join(DATA_DIR, temp, filename)
    orig_df = pd.read_csv(orig_path)
    orig_df = orig_df[orig_df['Type'] == ash_type].copy()
    orig_df = feature_engineering(orig_df)
    
    drop_cols = ['Type', 'group', 'Research paper', temp, 'Type_enc']
    X_orig = orig_df.drop(columns=[c for c in drop_cols if c in orig_df.columns])
    
    print(f"=== {feedstock} Ash ({temp}) Training Data Ranges ===")
    desc = X_orig.describe().T[['min', '25%', '50%', '75%', 'max', 'mean']]
    
    # New Data
    mask = new_df['Type'] == ash_type
    X_new = new_df[mask].copy()
    X_new = X_new.drop(columns=[c for c in drop_cols + ['Predicted_DT'] if c in X_new.columns])
    new_desc = X_new.describe().T[['min', 'max', 'mean']]
    
    # Compare
    comparison = pd.DataFrame({
        'Train_Min': desc['min'],
        'Train_Max': desc['max'],
        'New_Min': new_desc['min'],
        'New_Max': new_desc['max'],
    })
    
    # Detect out of range
    comparison['Out_Of_Range'] = (comparison['New_Min'] < comparison['Train_Min']) | (comparison['New_Max'] > comparison['Train_Max'])
    
    print(comparison)
    print("\n")
