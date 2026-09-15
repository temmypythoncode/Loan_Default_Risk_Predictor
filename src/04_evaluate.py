"""
04_evaluate.py
Evaluate both models properly. Accuracy alone is misleading on an
imbalanced target (a model that always predicts "no default" can still
score 70-80% accuracy while being useless) — so we look at precision,
recall, F1, and ROC-AUC, plus a confusion matrix and feature importance.
"""
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

OUTPUT_DIR = "outputs"

# --- Load models and test data -------------------------------------------------------
log_reg = joblib.load(f"{OUTPUT_DIR}/logistic_regression_model.pkl")
rf = joblib.load(f"{OUTPUT_DIR}/random_forest_model.pkl")
X_test = pd.read_csv(f"{OUTPUT_DIR}/X_test.csv")
X_test_scaled = pd.read_csv(f"{OUTPUT_DIR}/X_test_scaled.csv")
y_test = pd.read_csv(f"{OUTPUT_DIR}/y_test.csv").iloc[:, 0]


def evaluate_model(name, model, X, y_true):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved: {fname}")

    return y_proba, auc


# --- Evaluate both models -------------------------------------------------------
proba_lr, auc_lr = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)
proba_rf, auc_rf = evaluate_model("Random Forest", rf, X_test, y_test)

# --- ROC curve comparison -------------------------------------------------------
plt.figure(figsize=(6, 5))
for name, proba, auc in [("Logistic Regression", proba_lr, auc_lr), ("Random Forest", proba_rf, auc_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_comparison.png")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/roc_curve_comparison.png")

# --- Feature importance (Random Forest) -------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)
print("\n=== Top 10 features driving default risk (Random Forest) ===")
print(importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()
print(f"Saved: {OUTPUT_DIR}/feature_importance.png")

print("\nEvaluation complete. Check outputs/ for all plots and results.")
print("\nFor your writeup: compare ROC-AUC and recall on the 'Default' class")
print("between the two models — that's usually the headline comparison a")
print("capstone reviewer looks for.")
