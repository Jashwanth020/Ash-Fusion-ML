# Ash Fusion Temperature (AFT) Prediction Framework

This repository contains a complete, robust Machine Learning pipeline for predicting and analyzing the thermochemical Ash Fusion Temperatures (AFT) of Coal and Biomass feedstocks. 

Predicting Ash Fusion Temperatures is critical for mitigating slagging and fouling risks in industrial boilers and gasifiers. The framework targets all four standard ASTM fusion stages:
* **DT:** Deformation Temperature
* **ST:** Softening Temperature
* **HT:** Hemispherical Temperature
* **FT:** Flow Temperature

## Overview of the ML Pipeline

1. **Missing Data Imputation & Pre-processing:** Missing geochemical composition values are rigorously imputed via MICE (IterativeImputer with a Random Forest Regressor). Boundary conditions (e.g., '>1450', '>1500') are intelligently processed.
2. **Self-Adjustment (SA) Method:** A constrained boundary optimization technique is applied to dynamically filter out physically anomalous melting sequences, ensuring high-fidelity training datasets.
3. **Model Training & Hyperparameter Tuning:** State-of-the-art tree-based ensembles (**Extra Trees** and **XGBoost**) are trained and benchmarked. Hyperparameters are tuned utilizing Optuna Bayesian Optimization over 100 trials, validated through a $5 \times 3$ Repeated K-Fold cross-validation strategy.
4. **Monotonic Reconciliation (CLSO):** A post-processing Constrained Least Squares Optimization (CLSO) algorithm guarantees physical validity, ensuring predictions strictly obey thermodynamic melting laws ($DT \le ST \le HT \le FT$).
5. **Interpretability (SHAP):** Black-box predictions are demystified using Shapley Additive exPlanations (SHAP) and Permutation Importance, bridging the gap between statistical predictions and fundamental metallurgical fluxing mechanics.

## Repository Structure

* **`DATA/`**: Contains the raw and pre-processed (SA-filtered) composition data for both Coal and Biomass feedstocks.
* **`RESULTS/`**: Stores the finalized `.joblib` predictive models, detailed evaluation metrics (`metrics.csv`), and high-resolution visualization plots (actual vs. predicted scatter, residual analysis, SHAP beeswarm/bar/dependence plots).
* **`pipeline_*.py`**: The core modular Python scripts driving the data processing, imputation, training, and tuning for each respective temperature stage.
* **`reconcile.py`**: Implementation of the monotonic reconciliation algorithms (PAV and CLSO).
* **`predict_new_data.py`**: A dedicated inference script showcasing how to deploy the trained `.joblib` models on unseen experimental data (e.g., from XRF reports), applying MICE imputation and CLSO post-processing seamlessly.
* **`generate_*.py`**: A suite of plotting and visualization utilities utilized for feature attribution and model evaluation.

## Getting Started

To execute inference on new data:
1. Place your experimental XRF composition data into an Excel or CSV file.
2. Ensure the base oxide columns (`SiO2`, `Al2O3`, `Fe2O3`, `CaO`, `K2O`, etc.) are mapped correctly.
3. Run `predict_new_data.py` to automatically impute missing columns, engineer ratio features, predict AFTs, and apply physical reconciliation.

```bash
python predict_new_data.py
```

*Developed as part of an advanced machine learning investigation into non-linear phase equilibria governing fuel blending and thermochemical ash behavior.*
