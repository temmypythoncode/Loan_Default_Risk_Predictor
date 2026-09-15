"""
01_eda.py
Explore the loan dataset before doing anything else.
Goal: understand shape, missing values, class balance, and feature
distributions so later choices (encoding, scaling, model choice) are
informed rather than guessed.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---------------------------------------------------------------
DATA_PATH = "data/loan_data.csv"          # Kaggle: nikhil1e9/loan-default
FALLBACK_PATH = "data/sample_data.csv"    # used automatically if the above is missing
TARGET_COLUMN = "Default"
ID_COLUMNS = ["LoanID"]                   # identifier columns — drop before analysis
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load -------------------------------------------------------------
path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_PATH
print(f"Loading data from: {path}")
df = pd.read_csv(path)
df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

# --- Basic shape --------------------------------------------------------
print("\n=== Shape ===")
print(df.shape)

print("\n=== Column types ===")
print(df.dtypes)

print("\n=== First 5 rows ===")
print(df.head())

# --- Missing values -------------------------------------------------------
print("\n=== Missing values (count and %) ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_summary[missing_summary["missing_count"] > 0])

# --- Target / class balance -------------------------------------------------------
if TARGET_COLUMN not in df.columns:
    print(f"\n[!] TARGET_COLUMN '{TARGET_COLUMN}' not found. "
          f"Available columns: {list(df.columns)}")
    print("Update TARGET_COLUMN at the top of this file to match your dataset.")
else:
    print(f"\n=== Class balance ({TARGET_COLUMN}) ===")
    counts = df[TARGET_COLUMN].value_counts()
    pct = df[TARGET_COLUMN].value_counts(normalize=True).round(3) * 100
    print(pd.DataFrame({"count": counts, "pct": pct}))

    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET_COLUMN, data=df)
    plt.title("Class balance: Default vs No Default")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/class_balance.png")

# --- Numeric feature distributions -------------------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if TARGET_COLUMN in numeric_cols:
    numeric_cols.remove(TARGET_COLUMN)

if numeric_cols:
    df[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/numeric_distributions.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/numeric_distributions.png")

# --- Correlation heatmap (numeric only) -------------------------------------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/correlation_heatmap.png")

print("\nEDA complete. Check the outputs/ folder for plots.")
