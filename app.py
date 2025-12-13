import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
CLF_PATH = os.path.join(MODEL_DIR, "best_investment_classifier.pkl")
REG_PATH = os.path.join(MODEL_DIR, "best_future_price_regressor.pkl")

PREPROCESSED_FALLBACK = os.path.join(DATA_DIR, "india_housing_preprocessed.csv")
RAW_FALLBACK = os.path.join(DATA_DIR, "india_housing_prices.csv")


# ============================================================
# Helpers
# ============================================================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"metadata.json not found at: {METADATA_PATH}")
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"Classifier model not found at: {CLF_PATH}")
    if not os.path.exists(REG_PATH):
        raise FileNotFoundError(f"Regressor model not found at: {REG_PATH}")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return metadata, clf, reg


# ---------------- 
# - Keep ALL relevant cols needed for dropdown mapping + charts (including Price_in_Lakhs)
# - No drop columns like Facing/Security if they exist in dataset
# ----------------
@st.cache_data
def load_reference_data(feature_columns: list[str] | None = None):
    """
    Used to:
    - populate dependent dropdowns (State -> City -> Locality)
    - provide safe option lists for other categorical features
    - enable charts (City vs Price_in_Lakhs)
    """
    base_cols = [
        "State", "City", "Locality",
        "Property_Type", "Furnished_Status",
        "Public_Transport_Accessibility",
        "Facing", "Owner_Type", "Availability_Status",
        "Security", "Amenities"
    ]
    # Also include Price_in_Lakhs for insights chart if present
    numeric_cols = ["Price_in_Lakhs"]

    # If model expects more columns, include them too (safe)
    if feature_columns:
        wanted = sorted(set(base_cols + numeric_cols + feature_columns))
    else:
        wanted = sorted(set(base_cols + numeric_cols))

    for p in [PREPROCESSED_FALLBACK, RAW_FALLBACK]:
        if os.path.exists(p):
            try:
                df_ref = pd.read_csv(p)

                keep = [c for c in wanted if c in df_ref.columns]
                if keep:
                    df_ref = df_ref[keep].copy()

                # Clean text columns
                for c in df_ref.columns:
                    if df_ref[c].dtype == "object":
                        df_ref[c] = df_ref[c].astype(str).str.strip()

                # Drop invalid rows for mapping (State/City/Locality)
                if {"State", "City", "Locality"}.issubset(df_ref.columns):
                    df_ref = df_ref.dropna(subset=["State", "City", "Locality"])
                    for c in ["State", "City", "Locality"]:
                        df_ref = df_ref[df_ref[c].astype(str).str.lower() != "nan"]

                return df_ref
            except Exception:
                pass
    return None


def compute_defaults_from_data(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Build safe defaults for every feature column.
    Numeric -> median (or 0)
    Categorical -> mode (or "Unknown")
    """
    defaults = {}
    if df is None or df.empty:
        for col in feature_cols:
            defaults[col] = 0
        for col in feature_cols:
            if any(k in col.lower() for k in ["city", "state", "locality", "type", "status", "facing", "owner"]):
                defaults[col] = "Unknown"
        return defaults

    for col in feature_cols:
        if col not in df.columns:
            defaults[col] = "Unknown"
            continue

        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            val = s.replace([np.inf, -np.inf], np.nan).dropna()
            defaults[col] = float(val.median()) if len(val) else 0.0
        else:
            val = s.dropna()
            defaults[col] = str(val.mode().iloc[0]) if len(val) else "Unknown"

    return defaults


def sanitize_for_mlflow(params: dict) -> dict:
    """
    MLflow log_params wants scalars (string/float/int/bool).
    Convert lists/dicts safely to string.
    """
    clean = {}
    for k, v in params.items():
        if isinstance(v, (list, dict, tuple, set)):
            clean[k] = str(v)
        elif pd.isna(v):
            clean[k] = "NA"
        else:
            clean[k] = v
    return clean


def build_model_input(feature_cols: list[str], defaults: dict, user_inputs: dict) -> pd.DataFrame:
    """
    Create a single-row dataframe with EXACT feature columns expected by the model.
    Priority: user_inputs -> defaults -> 0/"Unknown"
    """
    row = {}
    for col in feature_cols:
        if col in user_inputs:
            row[col] = user_inputs[col]
        elif col in defaults:
            row[col] = defaults[col]
        else:
            row[col] = 0
    return pd.DataFrame([row], columns=feature_cols)


def try_get_feature_importance(pipeline):
    """
    Best-effort feature importance for RandomForest inside a Pipeline(preprocessor, model).
    Returns (feature_names, importances) or (None, None)
    """
    if not hasattr(pipeline, "named_steps"):
        return None, None

    if "model" not in pipeline.named_steps or "preprocessor" not in pipeline.named_steps:
        return None, None

    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return None, None

    try:
        feat_names = pre.get_feature_names_out()
        feat_names = [str(x) for x in feat_names]
    except Exception:
        feat_names = None

    return feat_names, model.feature_importances_


# ----------------
# - robust options helper (no "nan", no blanks)
# ----------------
def safe_unique(df, col, fallback):
    if df is not None and col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[(vals != "") & (vals.str.lower() != "nan")]
        out = sorted(vals.unique().tolist())
        if out:
            return out
    return fallback


# ----------------
# - probability should be 0..1 range, never crazy numbers
# ----------------
def get_probability_good_investment(model, X) -> float | None:
    # preferred path
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # choose class-1 column safely
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    idx = classes.index(1)
                elif "1" in classes:
                    idx = classes.index("1")
                else:
                    idx = 1 if proba.shape[1] > 1 else 0
            else:
                idx = 1 if proba.shape[1] > 1 else 0
            p = float(proba[0][idx])
            # clamp hard to avoid insane %
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            pass

    # fallback path
    if hasattr(model, "decision_function"):
        try:
            z = float(model.decision_function(X)[0])
            p = 1.0 / (1.0 + np.exp(-z))  # sigmoid
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            return None

    return None


# ============================================================
# App UI
# ============================================================
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.title("🏡 Real Estate Investment Advisor")
st.caption("Classification: Good Investment • Regression: Price After 5 Years • MLflow Tracking")

metadata, investment_model, price_model = load_artifacts()

feature_columns = metadata.get("feature_columns", [])
numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])

# FIX (4): pass feature_columns so reference df keeps more useful columns
ref_df = load_reference_data(feature_columns=feature_columns)
defaults = compute_defaults_from_data(ref_df, feature_columns)

st.sidebar.header("⚙️ Property Input Form")


# ----------------
# - Proper State -> City -> Locality mapping (dependent dropdowns)
# - City list is filtered by selected State
# - Locality list is filtered by selected City (within selected State)
# ----------------
user_inputs = {}

has_mapping = (
    ref_df is not None
    and {"State", "City", "Locality"}.issubset(ref_df.columns)
    and ("State" in feature_columns)
    and ("City" in feature_columns)
    and ("Locality" in feature_columns)
)

if has_mapping:
    state_options = safe_unique(ref_df, "State", ["Karnataka", "Tamil Nadu", "Maharashtra", "Delhi"])
    user_inputs["State"] = st.sidebar.selectbox("State", state_options, index=0)

    state_df = ref_df[ref_df["State"].astype(str).str.strip() == user_inputs["State"]]
    city_options = safe_unique(state_df, "City", ["Bangalore", "Chennai", "Mumbai", "Delhi"])
    user_inputs["City"] = st.sidebar.selectbox("City", city_options, index=0)

    city_df = state_df[state_df["City"].astype(str).str.strip() == user_inputs["City"]]
    locality_options = safe_unique(city_df, "Locality", ["Locality_1", "Locality_2"])
    user_inputs["Locality"] = st.sidebar.selectbox("Locality", locality_options, index=0)
else:
    # fallback (no dependent mapping possible)
    if "State" in feature_columns:
        state_opts = safe_unique(ref_df, "State", ["Karnataka", "Tamil Nadu", "Maharashtra", "Delhi"])
        user_inputs["State"] = st.sidebar.selectbox("State", state_opts, index=0)

    if "City" in feature_columns:
        cities = safe_unique(ref_df, "City", ["Bangalore", "Chennai", "Mumbai", "Delhi"])
        user_inputs["City"] = st.sidebar.selectbox("City", cities, index=0)

    if "Locality" in feature_columns:
        localities = safe_unique(ref_df, "Locality", ["Locality_1", "Locality_2"])
        user_inputs["Locality"] = st.sidebar.selectbox("Locality", localities, index=0)

# independent dropdowns
prop_types = safe_unique(ref_df, "Property_Type", ["Apartment", "Villa", "Independent House"])
furnish_vals = safe_unique(ref_df, "Furnished_Status", ["Unfurnished", "Semi-furnished", "Furnished"])
pta_vals = safe_unique(ref_df, "Public_Transport_Accessibility", ["Low", "Medium", "High"])

if "Property_Type" in feature_columns:
    user_inputs["Property_Type"] = st.sidebar.selectbox("Property Type", prop_types, index=0)

user_inputs["BHK"] = int(st.sidebar.slider("BHK", 1, 6, 3))
user_inputs["Size_in_SqFt"] = float(
    st.sidebar.number_input("Size (SqFt)", min_value=200.0, max_value=20000.0, value=1200.0, step=50.0)
)

if "Furnished_Status" in feature_columns:
    user_inputs["Furnished_Status"] = st.sidebar.selectbox("Furnishing Status", furnish_vals, index=0)

user_inputs["Floor_No"] = int(st.sidebar.slider("Floor No.", 0, 60, 2))
user_inputs["Total_Floors"] = int(st.sidebar.slider("Total Floors", 1, 80, 10))
user_inputs["Age_of_Property"] = int(st.sidebar.slider("Age of Property", 0, 100, 10))

user_inputs["Nearby_Schools"] = int(st.sidebar.slider("Nearby Schools", 0, 20, 3))
user_inputs["Nearby_Hospitals"] = int(st.sidebar.slider("Nearby Hospitals", 0, 20, 2))

if "Public_Transport_Accessibility" in feature_columns:
    user_inputs["Public_Transport_Accessibility"] = st.sidebar.selectbox(
        "Transport Access", pta_vals, index=min(2, len(pta_vals) - 1)
    )

# other optional model features
if "Facing" in feature_columns:
    face_opts = safe_unique(ref_df, "Facing", ["North", "South", "East", "West"])
    user_inputs["Facing"] = st.sidebar.selectbox("Facing", face_opts, index=0)

if "Parking_Space" in feature_columns:
    user_inputs["Parking_Space"] = int(st.sidebar.slider("Parking Space", 0, 5, 1))

if "Security" in feature_columns:
    user_inputs["Security"] = st.sidebar.selectbox("Security", ["Low", "Medium", "High"], index=1)

if "Amenities" in feature_columns:
    user_inputs["Amenities"] = st.sidebar.text_input("Amenities (comma-separated)", "Lift,Power Backup,Park")

if "Year_Built" in feature_columns:
    current_year = 2025
    user_inputs["Year_Built"] = int(
        st.sidebar.number_input(
            "Year Built",
            min_value=1900,
            max_value=current_year,
            value=max(1900, current_year - user_inputs["Age_of_Property"]),
        )
    )

if "Price_in_Lakhs" in feature_columns:
    user_inputs["Price_in_Lakhs"] = float(
        st.sidebar.number_input(
            "Current Price (Lakhs)",
            min_value=1.0,
            max_value=100000.0,
            value=float(defaults.get("Price_in_Lakhs", 250.0)),
            step=10.0,
        )
    )

# engineered column name consistency
if "Price_per_SqFt" in feature_columns:
    if "Price_in_Lakhs" in user_inputs and user_inputs["Size_in_SqFt"] > 0:
        user_inputs["Price_per_SqFt"] = float(user_inputs["Price_in_Lakhs"]) / float(user_inputs["Size_in_SqFt"])
    else:
        user_inputs["Price_per_SqFt"] = float(defaults.get("Price_per_SqFt", 0.0))

# Build final input_df matching model expectation
input_df = build_model_input(feature_columns, defaults, user_inputs)

st.subheader("📥 Input Summary (Model Features)")
st.dataframe(input_df, use_container_width=True)


# ============================================================
# MLflow setup
# ============================================================
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("real_estate_streamlit_app")


# ============================================================
# Prediction
# ============================================================
st.markdown("---")
run = st.button("Run Prediction")

if run:
    with mlflow.start_run():
        # ---------------- 
        # - prevent NameError issues (raw_inputs ALWAYS defined)
        # - log only user_inputs (not full feature row) for readability
        # ----------------
        raw_inputs = sanitize_for_mlflow(user_inputs)
        mlflow.log_params(raw_inputs)

        # ---- Classification
        inv_pred = investment_model.predict(input_df)[0]
        inv_prob = get_probability_good_investment(investment_model, input_df)
        if inv_prob is not None:
            mlflow.log_metric("investment_probability", inv_prob)

        # ---- Regression
        future_price = float(price_model.predict(input_df)[0])
        mlflow.log_metric("future_price_prediction", future_price)

        # ---- Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Investment Prediction")
            label = "Good Investment" if int(inv_pred) == 1 else "Not Recommended"
            st.success(label)

            # ---------------- 
            # - confidence is always % (0..100), never a crazy lakhs-like number
            # ----------------
            if inv_prob is not None:
                st.metric("Confidence (P[Good Investment])", f"{inv_prob * 100:.2f}%")
            else:
                st.info("Confidence unavailable (model has no predict_proba/decision_function).")

        with col2:
            st.subheader("📈 5-Year Price Forecast")
            st.metric("Estimated Future Price", f"₹ {future_price:,.2f} Lakhs")

        # ---- Feature importance (best-effort)
        st.markdown("---")
        st.subheader("🔍 Feature Importance (if available)")

        feat_names, importances = try_get_feature_importance(investment_model)
        if feat_names is not None and importances is not None:
            fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(20)

            fig, ax = plt.subplots()
            ax.barh(fi["feature"][::-1], fi["importance"][::-1])
            ax.set_title("Top 20 Feature Importances (Classifier)")
            ax.set_xlabel("Importance")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Feature importance not available for this model/pipeline.")

        # ---- Insights (data-driven)
        st.markdown("---")
        st.subheader("📊 Quick Market Insights (from dataset, if available)")

        if ref_df is None:
            st.info("No dataset found in /data to show insights. Add CSV to enable charts.")
        else:
            # City-wise median price
            if "City" in ref_df.columns and "Price_in_Lakhs" in ref_df.columns:
                tmp = ref_df.copy()
                tmp["Price_in_Lakhs"] = pd.to_numeric(tmp["Price_in_Lakhs"], errors="coerce")
                tmp = tmp.dropna(subset=["City", "Price_in_Lakhs"])
                city_med = tmp.groupby("City")["Price_in_Lakhs"].median().sort_values(ascending=False).head(15)

                fig, ax = plt.subplots()
                ax.bar(city_med.index.astype(str), city_med.values)
                ax.set_title("Top 15 Cities by Median Price (Lakhs)")
                ax.set_ylabel("Median Price (Lakhs)")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig, clear_figure=True)
            else:
                st.caption("Add 'Price_in_Lakhs' and 'City' columns in  r dataset to enable city price chart.")
