import os
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# ----------------------------
# CONFIG (Production-stable additions: 1), 2), 3)
# ----------------------------
DATA_PATH = os.path.join("data", "india_housing_preprocessed.csv")
PLOTS_DIR = "plots"
MODELS_DIR = "models"
MLRUNS_DIR = "mlruns"  # 3) ensure mlruns stored in project
MLFLOW_EXPERIMENT_NAME = "Real_Estate_Investment_Advisor"

# 2) Registry is often NOT supported on local file-based MLflow (Streamlit Cloud too)
ENABLE_REGISTRY = False  # set True only if using a real MLflow tracking server w/ registry support

# 3) Force MLflow tracking to a known location (project-local mlruns/)
mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS_DIR)}")


def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)


# ----------------------------
# MLflow logging (Classification)
# ----------------------------
def log_classification_run(
    model_name: str,
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any], str]:
    """
    Fit + evaluate a classification pipeline and log everything to MLflow.
    Returns trained pipeline, metrics dict, and run_id.
    """
    with mlflow.start_run(run_name=f"clf_{model_name}") as run:
        mlflow.log_param("task", "classification")
        mlflow.log_param("model_name", model_name)

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 5) ROC-AUC: check final estimator in pipeline (more reliable)
        roc_auc = None
        try:
            final_est = model_pipeline.named_steps.get("model", None)
            if final_est is not None and hasattr(final_est, "predict_proba"):
                y_prob = model_pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = None

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1_score", float(f1))
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", float(roc_auc))

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.csv")
        pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["True_0", "True_1"]).to_csv(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

        # Log full pipeline model
        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": (float(roc_auc) if roc_auc is not None else None),
        }

        return model_pipeline, metrics, run.info.run_id


# ----------------------------
# MLflow logging (Regression)
# ----------------------------
def log_regression_run(
    model_name: str,
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any], str]:
    """
    Fit + evaluate a regression pipeline and log everything to MLflow.
    Returns trained pipeline, metrics dict, and run_id.
    """
    with mlflow.start_run(run_name=f"reg_{model_name}") as run:
        mlflow.log_param("task", "regression")
        mlflow.log_param("model_name", model_name)

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        return model_pipeline, metrics, run.info.run_id


# ----------------------------
# OPTIONAL: Register best model to Model Registry
# ----------------------------
def register_best_model(run_id: str, model_name_in_registry: str):
    """
    Registers the 'model' artifact from a run into MLflow Model Registry.
    Creates a new version each time.

    NOTE: This often requires a real MLflow server + backend store (not local file store).
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name_in_registry)
    return mv


# ----------------------------
# MAIN TRAINING ENTRY POINT
# ----------------------------
def train_all_models_with_mlflow(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_clf: pd.Series,
    y_test_clf: pd.Series,
    y_train_reg: pd.Series,
    y_test_reg: pd.Series,
    classification_candidates: Dict[str, Pipeline],
    regression_candidates: Dict[str, Pipeline],
):
    """
    Runs all classification and regression candidates, logs to MLflow,
    and registers best classification + regression model (optional).
    """
    ensure_directories()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ----------------------------
    # Run classification experiments
    # ----------------------------
    clf_results = []
    for name, pipe in classification_candidates.items():
        trained_pipe, metrics, run_id = log_classification_run(
            name, pipe, X_train, X_test, y_train_clf, y_test_clf
        )
        clf_results.append((name, metrics, run_id, trained_pipe))

    # Choose best by F1-score (you can change criterion)
    clf_results_sorted = sorted(
        clf_results,
        key=lambda x: (x[1].get("f1_score") if x[1].get("f1_score") is not None else -1),
        reverse=True
    )
    best_clf_name, best_clf_metrics, best_clf_run_id, best_clf_pipe = clf_results_sorted[0]

    # Save best model locally
    best_clf_path = os.path.join(MODELS_DIR, "best_investment_classifier.pkl")
    joblib.dump(best_clf_pipe, best_clf_path)

    # 2) Register best classification model (optional, gated)
    if ENABLE_REGISTRY:
        register_best_model(best_clf_run_id, "Best_Investment_Classifier")

    # ----------------------------
    # Run regression experiments
    # ----------------------------
    reg_results = []
    for name, pipe in regression_candidates.items():
        trained_pipe, metrics, run_id = log_regression_run(
            name, pipe, X_train, X_test, y_train_reg, y_test_reg
        )
        reg_results.append((name, metrics, run_id, trained_pipe))

    # Choose best by RMSE (lower is better)
    reg_results_sorted = sorted(reg_results, key=lambda x: x[1]["rmse"])
    best_reg_name, best_reg_metrics, best_reg_run_id, best_reg_pipe = reg_results_sorted[0]

    # Save best model locally
    best_reg_path = os.path.join(MODELS_DIR, "best_future_price_regressor.pkl")
    joblib.dump(best_reg_pipe, best_reg_path)

    # 2) Register best regression model (optional, gated)
    if ENABLE_REGISTRY:
        register_best_model(best_reg_run_id, "Best_FuturePrice_Regressor")

    return {
        "best_classifier": {"name": best_clf_name, "metrics": best_clf_metrics, "run_id": best_clf_run_id},
        "best_regressor": {"name": best_reg_name, "metrics": best_reg_metrics, "run_id": best_reg_run_id},
        "saved_paths": {"classifier": best_clf_path, "regressor": best_reg_path},
    }


def create_targets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Good_Investment (binary) and Future_Price_5Y (numeric) if they are missing.
    Rule-based targets (can be tuned):
      - Future_Price_5Y: Price_in_Lakhs * (1 + 0.05 * 5)  => 25% 5-year growth
      - Good_Investment: 1 if expected ROI >= 20% (or adjust threshold)
    """

    # --- Find the base price column safely ---
    if "Price_in_Lakhs" not in df.columns:
        raise ValueError("Cannot create targets because 'Price_in_Lakhs' column is missing.")

    df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")

    # --- Create Future_Price_5Y if missing ---
    if "Future_Price_5Y" not in df.columns:
        annual_growth = 0.05     # 5% per year (tune)
        years = 5
        df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + annual_growth * years)

    # --- Create Good_Investment if missing ---
    if "Good_Investment" not in df.columns:
        roi_threshold = 0.20  # 20% total 5-year ROI (tune)
        expected_roi = (df["Future_Price_5Y"] - df["Price_in_Lakhs"]) / df["Price_in_Lakhs"]
        df["Good_Investment"] = (expected_roi >= roi_threshold).astype(int)

    return df



if __name__ == "__main__":
    ensure_directories()
    print("🚀 Starting MLflow training pipeline")

    # ----------------------------
    # 1) Load data
    # ----------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = create_targets_if_missing(df)

    # ----------------------------
    # 2) Validate required targets
    # ----------------------------
    required_targets = {"Good_Investment", "Future_Price_5Y"}
    missing = required_targets - set(df.columns)
    if missing:
        raise ValueError(f"❌ Still missing required target columns after creation: {missing}")

    # required_targets = {"Good_Investment", "Future_Price_5Y"}
    # missing = required_targets - set(df.columns)
    # if missing:
    #     raise ValueError(
    #         f"❌ Missing required target columns: {missing}\n"
    #         "Create these targets in preprocessing before training."
    #     )

    # 4) Make regression target numeric + drop NaNs in targets (production safety)
    df["Future_Price_5Y"] = pd.to_numeric(df["Future_Price_5Y"], errors="coerce")
    df = df.dropna(subset=["Good_Investment", "Future_Price_5Y"])

    # ----------------------------
    # 3) Split X/y
    # ----------------------------
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]

    X = df.drop(columns=["Good_Investment", "Future_Price_5Y"], errors="ignore")

    # Identify numeric vs categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # ----------------------------
    # 4) Preprocessing pipeline
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # ----------------------------
    # 5) Train/test split (same X split used for both tasks)
    # ----------------------------
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # For regression, align the split by index
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    # ----------------------------
    # 6) Candidate model pipelines
    # ----------------------------
    classification_candidates = {
        "logreg": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "rf_clf": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42))
        ]),
    }

    regression_candidates = {
        "linreg": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression())
        ]),
        "rf_reg": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42))
        ]),
    }

    # ----------------------------
    # 7) Run MLflow training + save best models
    # ----------------------------
    results = train_all_models_with_mlflow(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train_clf=y_train_clf,
        y_test_clf=y_test_clf,
        y_train_reg=y_train_reg,
        y_test_reg=y_test_reg,
        classification_candidates=classification_candidates,
        regression_candidates=regression_candidates,
    )

    print("✅ Training completed")
    print("📦 Saved model paths:", results["saved_paths"])
    print("🏆 Best Classifier:", results["best_classifier"]["name"], results["best_classifier"]["metrics"])
    print("🏆 Best Regressor:", results["best_regressor"]["name"], results["best_regressor"]["metrics"])
