import os
import pandas as pd
import numpy as np

DATA_DIR = r'p:\antigravity\DATA'

BEST_MODELS = {
    ('DT', 'Coal'): 'SA_ExtraTrees_combined_filtered.csv',
    ('DT', 'Biomass'): 'SA_ExtraTrees_combined_filtered.csv',
    ('FT', 'Coal'): 'SA_ExtraTrees_combined_filtered.csv',
    ('FT', 'Biomass'): 'SA_RandomForest_combined_filtered.csv',
    ('HT', 'Coal'): 'SA_XGBoost_combined_filtered.csv',
    ('HT', 'Biomass'): 'SA_ExtraTrees_combined_filtered.csv',
    ('ST', 'Coal'): 'SA_XGBoost_combined_filtered.csv',
    ('ST', 'Biomass'): 'SA_RandomForest_combined_filtered.csv',
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

results = []

for (temp, feedstock), filename in BEST_MODELS.items():
    ash_type = 'coal ash' if feedstock == 'Coal' else 'biomass ash'
    
    orig_path = os.path.join(DATA_DIR, temp, filename)
    if not os.path.exists(orig_path):
        continue
        
    orig_df = pd.read_csv(orig_path)
    orig_df = orig_df[orig_df['Type'] == ash_type].copy()
    orig_df = feature_engineering(orig_df)
    orig_df.dropna(subset=[temp], inplace=True)
    
    drop_cols = ['Type', 'group', 'Research paper', temp, 'Type_enc']
    X = orig_df.drop(columns=[c for c in drop_cols if c in orig_df.columns])
    
    desc = X.describe().T
    
    for feature in X.columns:
        results.append({
            'Model': f"{temp} {feedstock}",
            'Feedstock': feedstock,
            'Endpoint': temp,
            'Feature': feature,
            'Min': desc.loc[feature, 'min'],
            'Max': desc.loc[feature, 'max']
        })

df_res = pd.DataFrame(results)
df_res['Range'] = df_res.apply(lambda row: f"{row['Min']:.2f} - {row['Max']:.2f}", axis=1)

# Pivot tables
coal_pivot = df_res[df_res['Feedstock'] == 'Coal'].pivot(index='Feature', columns='Endpoint', values='Range')
biomass_pivot = df_res[df_res['Feedstock'] == 'Biomass'].pivot(index='Feature', columns='Endpoint', values='Range')

# Order rows
features_order = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'TiO2', 'SO3', 'Base_Acid_Ratio', 'Silica_Ratio', 'Alkali_Ratio']
coal_pivot = coal_pivot.reindex(features_order)
biomass_pivot = biomass_pivot.reindex(features_order)

print("### Coal Models Applicable Ranges (Min - Max %)")
print(coal_pivot.to_markdown())
print("\n### Biomass Models Applicable Ranges (Min - Max %)")
print(biomass_pivot.to_markdown())

