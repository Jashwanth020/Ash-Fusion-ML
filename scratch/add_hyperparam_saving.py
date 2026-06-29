import os
import re

file_path = r"P:\MVR\pipeline_core.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add json import if not exists
if "import json" not in content:
    content = content.replace("import optuna", "import optuna\nimport json")

# Find the spot to save the hyperparameters.
# It should be saved at the end of the experiment, near where we save the model.
old_save_model = """    # -- Save Best Model to Disk --
    model_path = os.path.join(results_dir, f"{best_model_name.replace(' ', '_')}_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\\n  [SAVE] Saved best model to: {model_path}")"""

new_save_model = """    # -- Save Best Model to Disk --
    model_path = os.path.join(results_dir, f"{best_model_name.replace(' ', '_')}_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\\n  [SAVE] Saved best model to: {model_path}")

    # -- Save Best Hyperparameters for all models --
    hyperparams_path = os.path.join(results_dir, 'best_hyperparameters.json')
    try:
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params_dict, f, indent=4)
        print(f"  [SAVE] Saved best hyperparameters to: {hyperparams_path}")
    except Exception as e:
        print(f"  [WARN] Failed to save hyperparameters: {e}")"""

if old_save_model in content:
    content = content.replace(old_save_model, new_save_model)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("pipeline_core.py updated to save hyperparameters.")
else:
    print("Could not find the target string to replace in pipeline_core.py")
