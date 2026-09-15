"""
05_predict_new_applicant.py
Score a single new loan applicant using the trained models.

Usage:
    Edit the `new_applicant` dictionary below with the applicant's details,
    then run:
        python src/05_predict_new_applicant.py

This mirrors the same cleaning + feature engineering + encoding steps used
in 02_preprocessing.py, so the applicant's data lines up with what the
models were trained on.
"""
import pandas as pd
import joblib
import os

OUTPUT_DIR = "outputs"

# --- Load trained models, scaler, and the training column layout -------------------------------------------------------
rf = joblib.load(f"{OUTPUT_DIR}/random_forest_model.pkl")
log_reg = joblib.load(f"{OUTPUT_DIR}/logistic_regression_model.pkl")
scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
training_columns = pd.read_csv(f"{OUTPUT_DIR}/processed_data.csv").drop(columns=["Default"]).columns

# ============================================================
# EDIT THIS: enter the new applicant's details here
# ============================================================
new_applicant = {
    "Age": 34,
    "Income": 48000,
    "LoanAmount": 18000,
    "CreditScore": 610,
    "MonthsEmployed": 42,
    "NumCreditLines": 3,
    "InterestRate": 14.5,
    "LoanTerm": 36,
    "DTIRatio": 0.40,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Single",
    "HasMortgage": "No",
    "HasDependents": "No",
    "LoanPurpose": "Debt Consolidation",
    "HasCoSigner": "No",
}
# ============================================================


def prepare_applicant(applicant_dict, training_columns):
    """Apply the same feature engineering + encoding used in training,
    then align columns so the row matches what the model expects."""
    df = pd.DataFrame([applicant_dict])

    # Same engineered feature as 02_preprocessing.py
    income_col = "Income" if "Income" in df.columns else "income" if "income" in df.columns else None
    loan_col = "LoanAmount" if "LoanAmount" in df.columns else "loan_amount" if "loan_amount" in df.columns else None
    if income_col and loan_col:
        df["loan_to_income_ratio"] = df[loan_col] / df[income_col].replace(0, 1)

    # One-hot encode the same way (drop_first=True, matching training)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align to training columns: add any missing dummy columns as 0,
    # drop anything extra, and put columns in the same order.
    df = df.reindex(columns=training_columns, fill_value=0)
    return df


X_new = prepare_applicant(new_applicant, training_columns)
X_new_scaled = scaler.transform(X_new)

rf_prob = rf.predict_proba(X_new)[0][1]
rf_pred = rf.predict(X_new)[0]

lr_prob = log_reg.predict_proba(X_new_scaled)[0][1]
lr_pred = log_reg.predict(X_new_scaled)[0]

print("=" * 50)
print("Loan Default Risk Assessment")
print("=" * 50)
print(f"\nApplicant: {new_applicant}\n")

print(f"Random Forest:        {'DEFAULT RISK' if rf_pred == 1 else 'Low risk'}  "
      f"(probability of default: {rf_prob:.1%})")
print(f"Logistic Regression:  {'DEFAULT RISK' if lr_pred == 1 else 'Low risk'}  "
      f"(probability of default: {lr_prob:.1%})")

avg_prob = (rf_prob + lr_prob) / 2
print(f"\nAverage risk estimate: {avg_prob:.1%}")

if avg_prob >= 0.5:
    print("=> High risk applicant. Recommend manual review before approval.")
elif avg_prob >= 0.3:
    print("=> Moderate risk. Consider stricter terms (higher rate, lower amount, or collateral).")
else:
    print("=> Low risk. Standard approval terms likely appropriate.")
