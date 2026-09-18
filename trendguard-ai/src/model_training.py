"""
MODULE 2 (Person B): MODEL TRAINING & MULTI-DISEASE PIPELINE
================================================================
Owns: training, model comparison, and saving ONE model PER DISEASE.
The same code trains diabetes and heart models -- nothing here is
disease-specific, which is what Gap 7 actually requires.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import get_train_test_split, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def get_candidate_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    return models


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def train_for_disease(disease_key: str):
    """Trains all candidate models for ONE disease and saves the best."""
    X_train, X_test, y_train, y_test, feature_names, raw_test = get_train_test_split(disease_key)

    best_model, best_score, best_name, results = None, -1, None, {}
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"[{disease_key}] {name}: {metrics}")
        if metrics["f1"] > best_score:
            best_score, best_model, best_name = metrics["f1"], model, name

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"best_model_{disease_key}.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, f"features_{disease_key}.pkl"))
    print(f"[{disease_key}] BEST: {best_name} (F1={best_score:.3f})\n")
    return best_model, best_name, results, feature_names


def train_all_diseases():
    """Called once by CI / setup: loops the SAME training code over
    every disease listed in config/datasets.yaml (Gap 7 in action)."""
    import yaml
    with open(os.path.join(ROOT, "config", "datasets.yaml")) as f:
        diseases = list(yaml.safe_load(f).keys())
    all_results = {}
    for d in diseases:
        _, name, results, _ = train_for_disease(d)
        all_results[d] = {"best_model": name, "metrics": results}
    return all_results


def load_model(disease_key: str):
    model_path = os.path.join(MODELS_DIR, f"best_model_{disease_key}.pkl")
    feat_path = os.path.join(MODELS_DIR, f"features_{disease_key}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model for '{disease_key}'. Run model_training.py first.")
    return joblib.load(model_path), joblib.load(feat_path)


def predict_risk(scaled_patient_row, disease_key: str):
    model, _ = load_model(disease_key)
    prob = model.predict_proba(scaled_patient_row)[0][1]
    pred_class = model.predict(scaled_patient_row)[0]
    return {"prediction": int(pred_class), "risk_percent": round(float(prob) * 100, 2)}


if __name__ == "__main__":
    train_all_diseases()
