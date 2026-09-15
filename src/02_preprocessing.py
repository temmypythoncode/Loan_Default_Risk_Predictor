"""
02_preprocessing.py
Clean the data and engineer features, then save a model-ready CSV.
"""
import pandas as pd
import numpy as np
import os

# --- Config ---------------------------------------------------------------
DATA_PATH = "data/loan_data.csv"
FALLBACK_PATH = "data/sample_data.csv"
TARGET_COLUMN = "Default"
ID_COLUMNS = ["LoanID"]
OUTPUT_DIR = "outputs"
OUTPUT_PATH = f"{OUTPUT_DIR}/processed_data.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load -------------------------------------------------------------
path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_PATH
print(f"Loading data from: {path}")
df = pd.read_csv(path)
df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"TARGET_COLUMN '{TARGET_COLUMN}' not found. "
        f"Available columns: {list(df.columns)}. "
        "Update TARGET_COLUMN at the top of this file."
    )

# --- Handle missing values -------------------------------------------------------
# Numeric columns: fill with median (robust to outliers).
# Categorical columns: fill with mode (most frequent value).
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if TARGET_COLUMN in numeric_cols:
    numeric_cols.remove(TARGET_COLUMN)
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in '{col}' with median ({median_val:.2f})")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled missing values in '{col}' with mode ('{mode_val}')")

# --- Feature engineering -------------------------------------------------------
# These ratios often carry more predictive signal than raw values alone.
# Works with either the real Kaggle columns (Income, LoanAmount) or the
# synthetic sample's lowercase columns (income, loan_amount).
income_col = "Income" if "Income" in df.columns else "income" if "income" in df.columns else None
loan_col = "LoanAmount" if "LoanAmount" in df.columns else "loan_amount" if "loan_amount" in df.columns else None

if income_col and loan_col:
    df["loan_to_income_ratio"] = df[loan_col] / df[income_col].replace(0, np.nan)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(0)
    print("Engineered: loan_to_income_ratio")

if "debt_to_income" not in df.columns and "DTIRatio" not in df.columns and income_col and loan_col:
    df["debt_to_income"] = df[loan_col] / df[income_col].replace(0, np.nan)
    df["debt_to_income"] = df["debt_to_income"].fillna(0)
    print("Engineered: debt_to_income")

# --- Encode categorical variables -------------------------------------------------------
# One-hot encoding: turns each category into its own 0/1 column.
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"\nOne-hot encoded: {categorical_cols}")

# --- Save -------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved processed data to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")
