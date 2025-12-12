"""
evaluate_best_models.py

Method 2: Evaluate ONLY the stored (best) models from /models and print metrics + markdown tables.

What it does:
1) Loads your dataset (preprocessed preferred, raw fallback).
2) Ensures targets exist (Good_Investment, Future_Price_5Y) — creates them if missing.
3) Recreates a reproducible train/test split (random_state=42).
4) Loads stored models:
   - models/best_investment_classifier.pkl
   - models/best_future_price_regressor.pkl
5) Evaluates and prints:
   - Classification: accuracy, precision, recall, f1, roc_auc (if possible), confusion matrix
   - Regression: rmse, mae, r2
6) Prints markdown tables (copy-paste into notebook/README) and saves a CSV report in ./reports

Run:
    python evaluate_best_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# ----------------------------
# CONFIG
# ----------------------------
PREPROCESSED_PATH = os.path.join("data", "india_housing_preprocessed.csv")  # preferred
RAW_PATH = os.path.join("data", "india_housing_prices.csv")                # fallback

CLF_PATH = os.path.join("models", "best_investment_classifier.pkl")
REG_PATH = os.path.join("models", "best_future_price_regressor.pkl")

REPORT_DIR = "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------
# Helpers
# ----------------------------
def load_dataset() -> pd.DataFrame:
    """Load preprocessed dataset if available, otherwise raw."""
    if os.path.exists(PREPROCESSED_PATH):
        print(f"✅ Loading preprocessed data: {PREPROCESSED_PATH}")
        return pd.read_csv(PREPROCESSED_PATH)
    if os.path.exists(RAW_PATH):
        print(f"✅ Loading raw data: {RAW_PATH}")
        return pd.read_csv(RAW_PATH)
    raise FileNotFoundError(
        f"❌ Could not find dataset.\nChecked:\n- {PREPROCESSED_PATH}\n- {RAW_PATH}"
    )


def create_targets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure targets exist. If missing, create them with a rule-based approach.

    Assumptions:
    - Uses Price_in_Lakhs as base price.
    - Future_Price_5Y = Price_in_Lakhs * (1 + annual_growth * years)
    - Good_Investment = 1 if ROI >= roi_threshold else 0
    """
    if "Price_in_Lakhs" not in df.columns:
        raise ValueError("❌ Cannot create targets because 'Price_in_Lakhs' column is missing.")

    df = df.copy()
    df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")

    # Create Future_Price_5Y
    if "Future_Price_5Y" not in df.columns:
        annual_growth = 0.05
        years = 5
        df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + annual_growth * years)

    # Create Good_Investment
    if "Good_Investment" not in df.columns:
        roi_threshold = 0.20
        expected_roi = (df["Future_Price_5Y"] - df["Price_in_Lakhs"]) / df["Price_in_Lakhs"]
        df["Good_Investment"] = (expected_roi >= roi_threshold).astype(int)

    # Clean targets
    df["Future_Price_5Y"] = pd.to_numeric(df["Future_Price_5Y"], errors="coerce")
    df["Good_Investment"] = pd.to_numeric(df["Good_Investment"], errors="coerce")

    df = df.dropna(subset=["Price_in_Lakhs", "Future_Price_5Y", "Good_Investment"])
    df["Good_Investment"] = df["Good_Investment"].astype(int)

    return df


def safe_predict_proba(model, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Try to compute probability for positive class (for ROC-AUC).
    Works for pipelines and estimators that support predict_proba.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
    except Exception:
        return None
    return None


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    y_prob = safe_predict_proba(model, X_test)
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def evaluate_regressor(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def to_markdown_table(d: Dict[str, Any], title: str) -> str:
    df = pd.DataFrame([d]).copy()
    # Convert non-scalars safely
    for col in df.columns:
        if isinstance(df[col].iloc[0], (list, dict)):
            df[col] = df[col].apply(lambda x: json.dumps(x))
    return f"\n## {title}\n\n" + df.to_markdown(index=False) + "\n"


# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load + prepare data
    df = load_dataset()
    df = create_targets_if_missing(df)

    # Targets
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]

    # Features
    X = df.drop(columns=["Good_Investment", "Future_Price_5Y"], errors="ignore")

    # 2) Split (reproducible)
    # IMPORTANT: classification split uses stratify; regression labels aligned by index
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
    )
    y_test_reg = y_reg.loc[X_test.index]

    # 3) Load stored models
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"❌ Missing classifier model file: {CLF_PATH}")
    if not os.path.exists(REG_PATH):
        raise FileNotFoundError(f"❌ Missing regressor model file: {REG_PATH}")

    print(f"✅ Loading classifier: {CLF_PATH}")
    clf_model = joblib.load(CLF_PATH)

    print(f"✅ Loading regressor: {REG_PATH}")
    reg_model = joblib.load(REG_PATH)

    # 4) Evaluate
    clf_metrics = evaluate_classifier(clf_model, X_test, y_test_clf)
    reg_metrics = evaluate_regressor(reg_model, X_test, y_test_reg)

    # 5) Print markdown for copy-paste
    clf_md = to_markdown_table({k: v for k, v in clf_metrics.items() if k != "confusion_matrix"}, "Stored Classifier Metrics")
    reg_md = to_markdown_table(reg_metrics, "Stored Regressor Metrics")

    print(clf_md)
    print(reg_md)

    # Optional: confusion matrix markdown
    cm = np.array(clf_metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["True_0", "True_1"])
    print("\n## Stored Classifier Confusion Matrix\n")
    print(cm_df.to_markdown())

    # 6) Save reports
    os.makedirs(REPORT_DIR, exist_ok=True)

    pd.DataFrame([{
        "model_file": os.path.basename(CLF_PATH),
        **{k: v for k, v in clf_metrics.items() if k != "confusion_matrix"}
    }]).to_csv(os.path.join(REPORT_DIR, "stored_classifier_metrics.csv"), index=False)

    pd.DataFrame([{
        "model_file": os.path.basename(REG_PATH),
        **reg_metrics
    }]).to_csv(os.path.join(REPORT_DIR, "stored_regressor_metrics.csv"), index=False)

    cm_df.to_csv(os.path.join(REPORT_DIR, "stored_classifier_confusion_matrix.csv"))

    # Save one combined markdown file
    with open(os.path.join(REPORT_DIR, "stored_models_metrics.md"), "w", encoding="utf-8") as f:
        f.write(clf_md)
        f.write("\n")
        f.write(reg_md)
        f.write("\n## Stored Classifier Confusion Matrix\n\n")
        f.write(cm_df.to_markdown())
        f.write("\n")

    print("\n✅ Saved reports:")
    print(f"- {os.path.join(REPORT_DIR, 'stored_classifier_metrics.csv')}")
    print(f"- {os.path.join(REPORT_DIR, 'stored_regressor_metrics.csv')}")
    print(f"- {os.path.join(REPORT_DIR, 'stored_classifier_confusion_matrix.csv')}")
    print(f"- {os.path.join(REPORT_DIR, 'stored_models_metrics.md')}")


if __name__ == "__main__":
    main()
