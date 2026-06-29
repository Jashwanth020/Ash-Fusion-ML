"""
generate_shap_waterfall.py — Generates SHAP Waterfall plots for individual predictions.

A waterfall plot explains a SINGLE prediction by showing how each feature 
pushed the model output from the base value (expected value over the dataset) 
to the final predicted value.

Usage:
    python generate_shap_waterfall.py --temp DT --ash-type Coal --sample-idx 0
    python generate_shap_waterfall.py --temp DT --ash-type Coal --best
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from pipeline_core import feature_engineering, MODEL_SA_MAP
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DATA_DIR = 'DATA'
RESULTS_DIR = 'RESULTS'

# We reuse the BEST_MODELS mapping from the global plots script
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

def find_best_model_path(temperature, ash_type):
    dir_path = os.path.join(RESULTS_DIR, temperature, ash_type)
    if not os.path.exists(dir_path):
        return None
    for file in os.listdir(dir_path):
        if file.endswith('.joblib'):
            return os.path.join(dir_path, file)
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate SHAP Waterfall Plot.')
    parser.add_argument('--temp', required=True, choices=['DT', 'ST', 'HT', 'FT'])
    parser.add_argument('--ash-type', required=True, choices=['Coal', 'Biomass'])
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of the sample to explain (default: 0)')
    parser.add_argument('--best', action='store_true', help='Automatically find and plot the sample with the most accurate prediction.')
    parser.add_argument('--output-dir', default='RESULTS/WATERFALL_PLOTS')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    model_path = find_best_model_path(args.temp, args.ash_type)
    if not model_path:
        print(f"[ERROR] Could not find .joblib model for {args.temp} {args.ash_type}")
        return
    
    print(f"  [INFO] Loading model: {os.path.basename(model_path)}")
    pipeline = joblib.load(model_path)
    tree_model = pipeline.named_steps['model']
    preprocessor = pipeline[:-1]

    # 2. Load Data
    model_name, filename = BEST_MODELS.get((args.temp, args.ash_type), (None, None))
    if not filename:
        print("[ERROR] Best model mapping not found.")
        return
        
    data_path = os.path.join(DATA_DIR, args.temp, filename)
    df = pd.read_csv(data_path)
    
    ash_type_str = 'coal ash' if args.ash_type == 'Coal' else 'biomass ash'
    df = df[df['Type'] == ash_type_str].copy()
    df = df.reset_index(drop=True)

    # 3. Feature Engineering
    df_engineered = feature_engineering(df)
    drop_cols = ['Type', 'group', 'Research paper', args.temp]
    X = df_engineered.drop(columns=[c for c in drop_cols if c in df_engineered.columns])
    y = df_engineered[args.temp].values
    feature_names = list(X.columns)

    # 4. Transform Data
    X_transformed = preprocessor.transform(X)

    # 5. Find the target sample index
    target_idx = args.sample_idx
    if args.best:
        print("  [INFO] Finding the best predicted sample...")
        y_pred = pipeline.predict(X)
        errors = np.abs(y - y_pred)
        target_idx = np.argmin(errors)
        best_error = errors[target_idx]
        print(f"  [FOUND] Best sample is Index {target_idx} with an absolute error of {best_error:.2f}°C")
        print(f"          (Actual: {y[target_idx]:.1f}°C, Predicted: {y_pred[target_idx]:.1f}°C)")
    else:
        if len(df) <= target_idx:
            print(f"[ERROR] Sample index {target_idx} is out of bounds for dataset of size {len(df)}")
            return
            
    print(f"\nGenerating SHAP Waterfall Plot for {args.temp} ({args.ash_type}) - Sample {target_idx}")

    # 6. Calculate SHAP
    explainer = shap.TreeExplainer(tree_model)
    try:
        explanation = explainer(X_transformed)
        if isinstance(explanation, list):
            explanation = explanation[0]
        explanation.data = X.values
        explanation.feature_names = feature_names
    except Exception as e:
        print(f"  [WARN] Native explainer(X) failed: {e}. Falling back to raw values.")
        shap_values_raw = explainer.shap_values(X_transformed)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
            shap_values_raw = shap_values_raw[0]
            
        explanation = shap.Explanation(
            values=shap_values_raw[target_idx],
            base_values=expected_value,
            data=X.values[target_idx],
            feature_names=feature_names
        )

    if not isinstance(explanation, shap.Explanation) or len(explanation.shape) > 1:
        sample_explanation = explanation[target_idx]
    else:
        sample_explanation = explanation

    # 7. Generate Plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sample_explanation, max_display=10, show=False)
    
    title = f"SHAP Waterfall: {args.temp} ({args.ash_type}) - Sample {target_idx}"
    if args.best:
        title += f"\n(Actual: {y[target_idx]:.0f}°C | Pred: {pipeline.predict(X.iloc[[target_idx]])[0]:.0f}°C)"
    plt.title(title, fontweight='bold', pad=20)
    
    suffix = "best" if args.best else f"sample{target_idx}"
    save_path = os.path.join(args.output_dir, f"waterfall_{args.temp}_{args.ash_type}_{suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [SUCCESS] Saved waterfall plot to {save_path}")

if __name__ == '__main__':
    main()
