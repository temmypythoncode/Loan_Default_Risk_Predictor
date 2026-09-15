"""
03_train_model.py
Train a baseline (Logistic Regression) and a stronger model (Random Forest),
using a stratified train/test split so the default rate is preserved in both.
"""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- Config ---------------------------------------------------------------
INPUT_PATH = "outputs/processed_data.csv"
TARGET_COLUMN = "Default"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load -------------------------------------------------------------
print(f"Loading processed data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in processed data.")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --- Train/test split -------------------------------------------------------
# stratify=y keeps the same default rate in both train and test sets —
# important because the target is imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train default rate: {y_train.mean():.3f}, Test default rate: {y_test.mean():.3f}")

# --- Scale numeric features (helps Logistic Regression converge properly) -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Baseline model: Logistic Regression -------------------------------------------------------
print("\n=== Training Logistic Regression (baseline) ===")
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
print("Done.")

# --- Stronger model: Random Forest -------------------------------------------------------
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)  # tree models don't need scaled features
print("Done.")

# --- Save everything needed for evaluation -------------------------------------------------------
joblib.dump(log_reg, f"{OUTPUT_DIR}/logistic_regression_model.pkl")
joblib.dump(rf, f"{OUTPUT_DIR}/random_forest_model.pkl")
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_scaled_df.to_csv(f"{OUTPUT_DIR}/X_test_scaled.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

print(f"\nModels and test data saved to: {OUTPUT_DIR}/")
print("Run 04_evaluate.py next.")
