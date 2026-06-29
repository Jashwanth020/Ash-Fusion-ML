"""
calculate_correlation.py — Compute and plot correlation matrices for Coal and Biomass ash samples.
Loads data from ash_data..xlsx, performs feature engineering, calculates Pearson correlation
coefficients, and exports the matrices as CSV files and high-resolution heatmaps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
XLSX_PATH = r"P:\antigravity\correlation\ash_data..xlsx"
OUTPUT_DIR = r"P:\antigravity\correlation"

RAW_OXIDES = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'P2O5', 'K2O', 'TiO2', 'SO3']
TEMPERATURES = ['DT', 'ST', 'HT', 'FT']

def feature_engineering(df):
    df = df.copy()
    
    # 1. Base/Acid Ratio
    denom_ba = df['SiO2'] + df['Al2O3'] + df['TiO2']
    # Replace zero with NaN to avoid division by zero
    denom_ba = denom_ba.replace(0, np.nan)
    df['Base_Acid_Ratio'] = (df['Fe2O3'] + df['CaO'] + df['MgO'] + df['Na2O'] + df['K2O']) / denom_ba
    
    # 2. Silica Ratio
    denom_si = df['SiO2'] + df['Fe2O3'] + df['CaO'] + df['MgO']
    denom_si = denom_si.replace(0, np.nan)
    df['Silica_Ratio'] = (100 * df['SiO2']) / denom_si
    
    # 3. Alkali Ratio
    df['Alkali_Ratio'] = df['Na2O'] + df['K2O']
    
    return df

def plot_correlation_heatmap(corr_df, title, save_path, figsize=(12, 10)):
    plt.figure(figsize=figsize, dpi=300)
    
    # Set style
    sns.set_theme(style="white")
    
    # Draw the heatmap
    cmap = "coolwarm" # Perceptually uniform diverging colormap for publication
    
    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
        annot_kws={"size": 10}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {save_path}")

def main():
    print(f"Loading data from: {XLSX_PATH}")
    xl = pd.ExcelFile(XLSX_PATH)
    
    # Check sheets
    print("Sheets in excel file:", xl.sheet_names)
    
    feedstocks = {
        'Coal': 'Coal Ash',
        'Biomass': 'Biomass Ash'
    }
    
    for label, sheet_name in feedstocks.items():
        print(f"\n==================================================")
        print(f"Processing Feedstock: {label} ({sheet_name})")
        print(f"==================================================")
        
        # Load and preprocess
        df = xl.parse(sheet_name)
        df = feature_engineering(df)
        
        # Define variable sets
        all_features = RAW_OXIDES + ['Base_Acid_Ratio', 'Silica_Ratio', 'Alkali_Ratio'] + TEMPERATURES
        raw_features = RAW_OXIDES + TEMPERATURES
        
        # Calculate correlation matrices
        corr_raw = df[raw_features].corr(method='pearson')
        corr_all = df[all_features].corr(method='pearson')
        
        # Save correlation values to CSV
        csv_raw_path = os.path.join(OUTPUT_DIR, f"{label.lower()}_correlation_raw.csv")
        csv_all_path = os.path.join(OUTPUT_DIR, f"{label.lower()}_correlation_all.csv")
        corr_raw.to_csv(csv_raw_path)
        corr_all.to_csv(csv_all_path)
        print(f"Saved correlation CSV (raw) to: {csv_raw_path}")
        print(f"Saved correlation CSV (all) to: {csv_all_path}")
        
        # Plot and save heatmaps
        raw_fig_path = os.path.join(OUTPUT_DIR, f"{label.lower()}_correlation_raw.png")
        all_fig_path = os.path.join(OUTPUT_DIR, f"{label.lower()}_correlation_all.png")
        
        plot_correlation_heatmap(
            corr_raw, 
            f"Correlation Matrix: {label} Ash (Raw Oxides & AFTs)", 
            raw_fig_path,
            figsize=(10, 8)
        )
        
        plot_correlation_heatmap(
            corr_all, 
            f"Correlation Matrix: {label} Ash (Oxides, Engineered Features & AFTs)", 
            all_fig_path,
            figsize=(12, 10)
        )
        
        # Print key correlations with temperatures
        print(f"\nCorrelation of features with Temperatures ({label} Ash):")
        temp_corr = corr_all.loc[TEMPERATURES, RAW_OXIDES + ['Base_Acid_Ratio', 'Silica_Ratio', 'Alkali_Ratio']].T
        print(temp_corr.round(3))

if __name__ == '__main__':
    main()
