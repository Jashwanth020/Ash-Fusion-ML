import os
import re

file_path = r"P:\MVR\pipeline_core.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add joblib import
if "import joblib" not in content:
    content = content.replace("import optuna", "import optuna\nimport joblib")

# 2. Modify run_full_pipeline
old_full_pipeline = """    # Load consensus data
    df_consensus = load_consensus_data(temperature, data_dir)
    df_consensus.drop(columns=['group', 'Research paper'],
                      errors='ignore', inplace=True)
    df_consensus.dropna(subset=[temperature, 'Type'], inplace=True)

    # Run for each ash type
    results = {}
    for ash_type in ['coal ash', 'biomass ash']:
        result = run_experiment(
            temperature=temperature,
            ash_type=ash_type,
            df_consensus=df_consensus,
            data_dir=data_dir,
            results_base=results_base,
            quick=quick,
        )"""

new_full_pipeline = """    # Run for each ash type (No consensus data available)
    results = {}
    for ash_type in ['coal ash', 'biomass ash']:
        result = run_experiment(
            temperature=temperature,
            ash_type=ash_type,
            data_dir=data_dir,
            results_base=results_base,
            quick=quick,
        )"""

content = content.replace(old_full_pipeline, new_full_pipeline)

# 3. Modify run_experiment signature
content = content.replace(
    "def run_experiment(temperature, ash_type, df_consensus, data_dir='DATA',",
    "def run_experiment(temperature, ash_type, data_dir='DATA',"
)

# 4. Remove consensus data usage in run_experiment
# Find the start of the consensus block
start_idx = content.find("    # -- Prepare consensus data for this ash type --")
# Find the end of the correlation heatmap block
end_idx = content.find("    # -- Train each model on its SA-filtered data --")

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + content[end_idx:]

# 5. Fix "Ensure same features as consensus"
old_feature_align = """        # Ensure same features as consensus
        # (SA files might have same columns but different rows)
        missing_feats = set(feature_names) - set(X_sa.columns)
        extra_feats = set(X_sa.columns) - set(feature_names)
        if missing_feats:
            for f in missing_feats:
                X_sa[f] = np.nan
        if extra_feats:
            X_sa = X_sa.drop(columns=list(extra_feats))
        X_sa = X_sa[feature_names]"""

# Just remove this block because feature_names is not defined from consensus anymore.
# Wait, feature_names was defined from consensus. I need to define it from X_sa!
new_feature_align = """        feature_names = list(X_sa.columns)"""
content = content.replace(old_feature_align, new_feature_align)


# 6. Replace Stacking Ensemble with Model Saving
old_stacking = """    # -- Stacking Ensemble (on consensus data) --
    print(f"\\n  -- Stacking Ensemble (Bypassed) --")
    try:
        pass
    except Exception as e:
        print(f"  [WARN] Stacking ensemble failed: {e}")"""

new_stacking = """    # -- Save Best Model to Disk --
    model_path = os.path.join(results_dir, f"{best_model_name.replace(' ', '_')}_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\\n  [SAVE] Saved best model to: {model_path}")"""

content = content.replace(old_stacking, new_stacking)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("pipeline_core.py updated successfully.")
