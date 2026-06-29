"""
predict_new_data.py — Inference script specifically for the XRF_Analysis_Report.xlsx
Includes MICE imputation for the missing Na2O column trained on the original Coal data.
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pipeline_core import feature_engineering
from reconcile import reconcile

warnings.filterwarnings('ignore')

RESULTS_DIR = 'RESULTS'
TEMPERATURES = ['DT', 'ST', 'HT', 'FT']
ASH_TYPE = 'Coal'

COL_MAP = {
    'SiO₂': 'SiO2', 'Al₂O₃': 'Al2O3', 'Fe₂O₃': 'Fe2O3', 
    'K₂O': 'K2O', 'CaO': 'CaO', 'MgO': 'MgO', 
    'P₂O₅': 'P2O5', 'TiO₂': 'TiO2', 'SO₃': 'SO3'
}

BASE_COLS = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'P2O5', 'K2O', 'TiO2', 'SO3']

def find_best_model_path(temperature, ash_type):
    dir_path = os.path.join(RESULTS_DIR, temperature, ash_type)
    if not os.path.exists(dir_path):
        return None
    for file in os.listdir(dir_path):
        if file.endswith('.joblib'):
            return os.path.join(dir_path, file)
    return None

def get_model_rmse(temperature, ash_type, results_dir='RESULTS'):
    csv_path = os.path.join(results_dir, temperature, ash_type, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        model_path = find_best_model_path(temperature, ash_type, results_dir)
        if not model_path: return None
        model_name_file = os.path.basename(model_path).replace('_model.joblib', '').replace('_', ' ')
        row = df[df['Model'] == model_name_file]
        if not row.empty:
            return float(row.iloc[0]['Test_RMSE'])
    except Exception as e:
        print(f"  [WARN] Failed to read RMSE for {temperature}: {e}")
    return None

def main():
    print("=" * 60)
    print("  PREDICTING ON XRF_Analysis_Report.xlsx (Coal Models)")
    print("=" * 60)

    # 1. Train MICE Imputer on original training data
    print("[INFO] Training MICE imputer on original Coal data...")
    # Using one of the SA datasets as the base representative dataset for training the imputer
    train_df = pd.read_csv(r'DATA\DT\SA_ExtraTrees_combined_filtered.csv')
    train_df = train_df[train_df['Type'] == 'coal ash'][BASE_COLS].copy()
    
    mice_imputer = IterativeImputer(random_state=42, max_iter=10)
    mice_imputer.fit(train_df)

    # 2. Read new Excel data
    excel_path = r'P:\MVR\new_data_exp\XRF_Analysis_Report.xlsx'
    df_raw = pd.read_excel(excel_path, header=0)
    
    # Keep track of sample names
    sample_names = df_raw['Blend Sample'].copy()
    
    df = df_raw.rename(columns=COL_MAP).copy()
    
    # Add missing Na2O column as NaN
    df['Na2O'] = np.nan
    
    # Extract only the base oxide columns for imputation
    X_to_impute = df[BASE_COLS].copy()

    # 3. Impute Na2O
    print("[INFO] Imputing missing Na2O values using MICE...")
    X_imputed_array = mice_imputer.transform(X_to_impute)
    X_imputed = pd.DataFrame(X_imputed_array, columns=BASE_COLS, index=df.index)
    
    # Update the dataframe with imputed values
    for col in BASE_COLS:
        df[col] = X_imputed[col]

    # Print the imputed Na2O values so the user can see them
    print("\n--- Imputed Na2O Values ---")
    for idx, name in enumerate(sample_names):
        print(f"  {name}: {df.loc[idx, 'Na2O']:.4f}%")

    # 4. Feature Engineering
    print("\n[INFO] Applying feature engineering...")
    df_engineered = feature_engineering(df)

    # 5. Predict using best models
    predictions = pd.DataFrame(index=df.index)
    weights = []
    
    for temp in TEMPERATURES:
        model_path = find_best_model_path(temp, ASH_TYPE)
        if not model_path:
            print(f"[ERROR] Could not find .joblib model for {temp} ({ASH_TYPE})")
            return
            
        print(f"[INFO] Predicting {temp} using {os.path.basename(model_path)}")
        pipeline = joblib.load(model_path)
        
        # Ensure we only pass the columns the pipeline expects
        try:
            features_in = pipeline.named_steps['imputer'].feature_names_in_
            X_model = df_engineered[features_in]
        except AttributeError:
            X_model = df_engineered
            
        predictions[temp] = pipeline.predict(X_model)
        
        rmse = get_model_rmse(temp, ASH_TYPE)
        if rmse is not None and rmse > 0:
            weights.append(1.0 / (rmse ** 2))
        else:
            weights.append(1.0)

    print("\n--- RAW PREDICTIONS ---")
    print(predictions)
    
    print("\n--- WEIGHTS (Inverse Variance) ---")
    for t, w in zip(TEMPERATURES, weights):
        print(f"  {t}: {w:.6f}")

    # 6. Apply Reconciliation
    print("\n[INFO] Applying Monotonic Reconciliation...")
    reconciled_preds = reconcile(predictions, verbose=True, weights=weights)

    # 7. Final Assembly & Export
    final_output = pd.concat([df_raw, reconciled_preds], axis=1)
    
    out_path = r'P:\MVR\new_data_exp\Predicted_AFT_Reconciled.csv'
    final_output.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Final predictions exported to: {out_path}")

if __name__ == '__main__':
    main()
