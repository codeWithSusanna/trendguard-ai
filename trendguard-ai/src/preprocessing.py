"""
MODULE 1 (Person A): DATA & TREND ENGINEERING
================================================
Owns: config/datasets.yaml loading, cleaning, TREND FEATURE
ENGINEERING (Gap 1), and multi-dataset support (Gap 7).

Every other module calls get_train_test_split(disease_key) and never
touches raw CSVs directly.
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "datasets.yaml")
MODELS_DIR = os.path.join(ROOT, "models")


def load_config(disease_key: str) -> dict:
    with open(CONFIG_PATH) as f:
        all_config = yaml.safe_load(f)
    if disease_key not in all_config:
        raise ValueError(f"Unknown disease_key '{disease_key}'. "
                          f"Available: {list(all_config.keys())}")
    return all_config[disease_key]


def _synthetic_dataset(disease_key: str, cfg: dict) -> pd.DataFrame:
    """Fallback demo data generator so the pipeline runs even before
    the real Kaggle CSV is downloaded. Used for BOTH diabetes and
    heart datasets, driven entirely by the config -- this is what
    keeps the pipeline generic across diseases (Gap 7)."""
    rng = np.random.default_rng(7)
    n = 400
    if disease_key == "diabetes":
        df = pd.DataFrame({
            "Pregnancies": rng.integers(0, 10, n),
            "Glucose": rng.normal(120, 30, n).clip(0, 250),
            "BloodPressure": rng.normal(70, 15, n).clip(0, 140),
            "SkinThickness": rng.normal(20, 10, n).clip(0, 60),
            "Insulin": rng.normal(80, 60, n).clip(0, 500),
            "BMI": rng.normal(32, 7, n).clip(10, 60),
            "DiabetesPedigreeFunction": rng.normal(0.5, 0.3, n).clip(0.05, 2.5),
            "Age": rng.integers(21, 81, n),
        })
        score = 0.03 * df["Glucose"] + 0.02 * df["BMI"] + 0.01 * df["Age"] - 4
    else:  # heart
        df = pd.DataFrame({
            "age": rng.integers(29, 77, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.normal(130, 17, n).clip(90, 200),
            "chol": rng.normal(246, 50, n).clip(120, 450),
            "thalach": rng.normal(150, 22, n).clip(70, 200),
            "oldpeak": rng.normal(1.0, 1.1, n).clip(0, 6),
            "ca": rng.integers(0, 4, n),
        })
        score = 0.02 * df["chol"] + 0.015 * df["age"] - df["thalach"] * 0.01 - 4

    prob = 1 / (1 + np.exp(-((score - score.mean()) / score.std())))
    df[cfg["target"]] = (rng.random(n) < prob).astype(int)
    return df


def load_raw_data(disease_key: str, cfg: dict) -> pd.DataFrame:
    path = os.path.join(ROOT, cfg["path"])
    if os.path.exists(path):
        print(f"[preprocessing] Loaded real '{disease_key}' dataset from {path}")
        return pd.read_csv(path)
    print(f"[preprocessing] WARNING: {path} not found. Using synthetic "
          f"'{disease_key}' sample data for pipeline testing. Download the "
          f"real Kaggle CSV before final submission.")
    return _synthetic_dataset(disease_key, cfg)


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    for col in cfg.get("zero_as_missing", []):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df.drop_duplicates().dropna()


# ---------------------------------------------------------------
# GAP 1: TREND FEATURE ENGINEERING
# ---------------------------------------------------------------
def add_trend_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Turns repeated-visit data into trend features (change over time),
    which is the whole point of Gap 1: most public models only ever
    see a single snapshot per patient.

    Two modes:
    1. REAL longitudinal data: if patient_id_col + visit_col exist in
       the dataframe, pivot per patient and compute real slopes.
    2. DEMO mode: if the dataset is single-visit (like the public Pima
       CSV), we simulate two earlier visits per patient using small
       controlled noise, purely so the trend-feature *pipeline* can be
       demonstrated end-to-end. This is clearly logged and must be
       replaced with real multi-visit data for genuine research use.
    """
    trend_cols = [c for c in cfg.get("trend_features", []) if c in df.columns]
    if not trend_cols:
        return df

    id_col = cfg.get("patient_id_col")
    visit_col = cfg.get("visit_col")

    if id_col in df.columns and visit_col in df.columns:
        return _real_longitudinal_trends(df, id_col, visit_col, trend_cols)

    print("[preprocessing] No multi-visit columns found -> simulating "
          "2 earlier checkups per patient to DEMO the trend pipeline. "
          "Replace with real longitudinal data for the final project.")
    return _simulated_trends(df, trend_cols)


def _real_longitudinal_trends(df, id_col, visit_col, trend_cols):
    df_sorted = df.sort_values([id_col, visit_col])
    out_rows = []
    for pid, grp in df_sorted.groupby(id_col):
        grp = grp.sort_values(visit_col)
        latest = grp.iloc[-1].copy()
        for col in trend_cols:
            series = grp[col].values
            if len(series) >= 2:
                latest[f"{col}_change"] = series[-1] - series[0]
                latest[f"{col}_trend_slope"] = np.polyfit(range(len(series)), series, 1)[0]
            else:
                latest[f"{col}_change"] = 0.0
                latest[f"{col}_trend_slope"] = 0.0
        out_rows.append(latest)
    return pd.DataFrame(out_rows).reset_index(drop=True)


def _simulated_trends(df, trend_cols, rng_seed=11):
    rng = np.random.default_rng(rng_seed)
    df = df.copy()
    for col in trend_cols:
        current = df[col].values
        visit1 = current * (1 - rng.normal(0.06, 0.03, len(df)))  # ~6% lower, 2 checkups ago
        visit2 = current * (1 - rng.normal(0.03, 0.02, len(df)))  # ~3% lower, 1 checkup ago
        df[f"{col}_change"] = current - visit1
        # slope across the 3 simulated points (visit1 -> visit2 -> current)
        slopes = []
        for v1, v2, v3 in zip(visit1, visit2, current):
            slopes.append(np.polyfit([0, 1, 2], [v1, v2, v3], 1)[0])
        df[f"{col}_trend_slope"] = slopes
    return df


def get_train_test_split(disease_key: str, test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: config -> load -> clean -> trend features -> scale -> split.
    Returns X_train, X_test, y_train, y_test, feature_names, raw_test_df
    (raw_test_df is kept, unscaled, so Module 3 can run fairness checks
    against real column values like Age/sex).
    """
    cfg = load_config(disease_key)
    df = load_raw_data(disease_key, cfg)
    df = clean_data(df, cfg)
    df = add_trend_features(df, cfg)

    drop_cols = [c for c in [cfg.get("patient_id_col"), cfg.get("visit_col")] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{disease_key}.pkl"))

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names, X_test.reset_index(drop=True)


def preprocess_single_patient(patient_dict: dict, feature_names: list, disease_key: str) -> np.ndarray:
    """Used by Module 4 (UI) at inference time."""
    scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{disease_key}.pkl"))
    row = [patient_dict.get(f, 0) for f in feature_names]
    return scaler.transform([row])


if __name__ == "__main__":
    for disease in ["diabetes", "heart"]:
        X_train, X_test, y_train, y_test, feature_names, raw_test = get_train_test_split(disease)
        print(f"\n[{disease}] features: {feature_names}")
        print(f"[{disease}] train shape: {X_train.shape}, test shape: {X_test.shape}")
