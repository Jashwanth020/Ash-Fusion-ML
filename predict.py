"""
predict.py — Inference script for Ash Fusion Temperatures

Loads trained models, predicts on new data, and automatically 
applies reconciliation (DT <= ST <= HT <= FT).

Usage:
    python predict.py --input new_data.csv --ash-type "Coal" --output results.csv
"""

import os
import argparse
import joblib
import pandas as pd
from pipeline_core import feature_engineering, MODEL_NAMES
from reconcile import reconcile

# You must manually map the BEST model for each temperature based on your training runs.
# Since we don't have a centralized registry, we'll try to find any saved model.
TEMPERATURES = ['DT', 'ST', 'HT', 'FT']

def find_best_model_path(temperature, ash_type, results_dir='RESULTS'):
    """Finds the first .joblib file in the results directory for the given temp/ash."""
    dir_path = os.path.join(results_dir, temperature, ash_type)
    if not os.path.exists(dir_path):
        return None
    for file in os.listdir(dir_path):
        if file.endswith('.joblib'):
            return os.path.join(dir_path, file)
    return None

def get_model_rmse(temperature, ash_type, results_dir='RESULTS'):
    """Reads the Test_RMSE of the best model from metrics.csv."""
    csv_path = os.path.join(results_dir, temperature, ash_type, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        # Find the row for the best model (assume it's the one with highest Test_R2 or just take the top one if sorted, 
        # wait, the model name is in the joblib file, but let's just find the minimum Test_RMSE in the file 
        # or the one corresponding to the best model. 
        # Actually, let's just read the joblib model name)
        # For simplicity, we just find the best model path, get its name, and look it up.
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
    parser = argparse.ArgumentParser(description='Predict and reconcile AFT.')
    parser.add_argument('--input', required=True, help='Path to input CSV containing new samples.')
    parser.add_argument('--ash-type', required=True, choices=['Coal', 'Biomass'], help='Ash type (Coal or Biomass).')
    parser.add_argument('--output', default='reconciled_predictions.csv', help='Path to output CSV.')
    args = parser.parse_args()

    print("=" * 60)
    print("  ASH FUSION TEMPERATURE INFERENCE")
    print("=" * 60)

    # 1. Load Data
    print(f"[INFO] Loading input data from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {args.input}")
        return

    # 2. Feature Engineering
    # The models expect 'Base_Acid_Ratio', 'Silica_Ratio', 'Alkali_Ratio'
    print(f"[INFO] Applying feature engineering...")
    df = feature_engineering(df)
    
    # Extract features matching the model's training data
    # (assuming all standard oxides are present in the input CSV)
    # The pipeline uses a RobustScaler and SimpleImputer internally, so missing values are fine.

    # 3. Load Models and Predict
    predictions = pd.DataFrame(index=df.index)
    weights = []
    
    for temp in TEMPERATURES:
        model_path = find_best_model_path(temp, args.ash_type)
        if not model_path:
            print(f"[ERROR] No trained model found for {temp} ({args.ash_type}). Run pipelines first!")
            return
            
        print(f"[INFO] Loading {temp} model: {os.path.basename(model_path)}")
        pipeline = joblib.load(model_path)
        
        # We need to extract the feature names the pipeline expects
        # Depending on scikit-learn version, pipeline might not have feature_names_in_
        # But we can pass the whole dataframe and let the imputer/scaler handle it if columns align.
        # It's safer to pass exactly the columns it expects.
        try:
            # Try to get feature names from the imputer
            features_in = pipeline.named_steps['imputer'].feature_names_in_
            X = df[features_in]
        except AttributeError:
            # Fallback: Just pass the engineered df (might fail if extra/missing columns)
            print(f"  [WARN] Could not strictly align features for {temp}. Ensure CSV matches training data.")
            X = df
            
        print(f"  [PREDICT] Generating {temp} predictions...")
        predictions[temp] = pipeline.predict(X)
        
        # Get RMSE for weighting
        rmse = get_model_rmse(temp, args.ash_type)
        if rmse is not None and rmse > 0:
            weights.append(1.0 / (rmse ** 2))
        else:
            weights.append(1.0) # Fallback uniform weight

    print("\n--- RAW PREDICTIONS ---")
    print(predictions.head())
    
    print("\n--- WEIGHTS (Inverse Variance) ---")
    for t, w in zip(TEMPERATURES, weights):
        print(f"  {t}: {w:.6f}")

    # 4. Reconcile
    print("\n[INFO] Applying Monotonic Reconciliation (PAV & CLSO)...")
    reconciled_df = reconcile(predictions, verbose=True, weights=weights)

    # 5. Save Output
    # Combine original data with predictions
    final_output = pd.concat([df, reconciled_df], axis=1)
    final_output.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Saved reconciled predictions to {args.output}")


if __name__ == '__main__':
    main()
