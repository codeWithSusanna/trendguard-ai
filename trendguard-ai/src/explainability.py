"""
MODULE 3 (Person C): EXPLAINABILITY + FAIRNESS
=================================================
Owns TWO gaps:
  - Gap 2: SHAP explanations cross-checked against known clinical
    risk factors, with an explicit correlation-vs-causation note.
  - Gap 6: subgroup fairness audit (does the model perform worse for
    some age/gender group?).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_training import load_model
from preprocessing import get_train_test_split, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------
# GAP 2: SHAP + domain-knowledge cross-check
# ---------------------------------------------------------------
def explain_with_shap(patient_row_scaled, feature_names, disease_key, background_data=None):
    import shap
    model, _ = load_model(disease_key)

    if background_data is None:
        X_train, _, _, _, _, _ = get_train_test_split(disease_key)
        background_data = X_train[:100]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_row_scaled)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(patient_row_scaled, nsamples=100)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = dict(zip(feature_names, values))
    sorted_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    plt.figure(figsize=(7, 5))
    feats, vals = list(sorted_contrib.keys()), list(sorted_contrib.values())
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    plt.barh(feats[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("SHAP value (impact on risk prediction)")
    plt.title(f"Why this prediction? ({disease_key})")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"shap_{disease_key}.png")
    plt.savefig(save_path)
    plt.close()

    return sorted_contrib, save_path


def cross_check_with_domain_knowledge(contributions: dict, disease_key: str, top_n: int = 4) -> dict:
    """
    GAP 2 CORE FUNCTION.
    Compares SHAP's top-N most influential features for THIS patient
    against the clinically established top risk factors (defined in
    config/datasets.yaml -> domain_top_factors).

    Returns a report explaining overlap/mismatch, plus the mandatory
    correlation-vs-causation disclaimer -- this is the actual
    "how we filled Gap 2" deliverable, not just running SHAP.
    """
    cfg = load_config(disease_key)
    domain_factors = set(cfg.get("domain_top_factors", []))

    model_top = list(contributions.keys())[:top_n]
    overlap = [f for f in model_top if f in domain_factors]
    not_in_domain_list = [f for f in model_top if f not in domain_factors]

    agreement_pct = round(100 * len(overlap) / max(len(model_top), 1), 1)

    return {
        "model_top_factors": model_top,
        "known_clinical_factors": list(domain_factors),
        "overlap": overlap,
        "flagged_for_review": not_in_domain_list,
        "agreement_percent": agreement_pct,
        "disclaimer": (
            "SHAP values show statistical correlation learned by the model, "
            "NOT proven medical causation. Features flagged here as "
            "'not in the known clinical list' are not necessarily wrong -- "
            "they should be reviewed by a domain expert, not treated as "
            "a mistake."
        ),
    }


def get_top_reasons(contributions: dict, top_n: int = 3) -> str:
    top_items = list(contributions.items())[:top_n]
    reasons = [f"{feat} ({'increased' if val > 0 else 'decreased'} risk)" for feat, val in top_items]
    return "Top contributing factors: " + ", ".join(reasons)


# ---------------------------------------------------------------
# GAP 6: Subgroup fairness audit
# ---------------------------------------------------------------
def evaluate_subgroup_fairness(disease_key: str) -> dict:
    """
    Splits the test set into subgroups (e.g. Age bands, sex) as defined
    in config/datasets.yaml -> fairness_attrs, and reports accuracy/
    recall PER GROUP. A model that looks great overall but performs
    much worse on one group is exactly the kind of bias Gap 6 flags.
    """
    from sklearn.metrics import accuracy_score, recall_score

    cfg = load_config(disease_key)
    model, feature_names = load_model(disease_key)
    X_train, X_test, y_train, y_test, feat_names, raw_test_df = get_train_test_split(disease_key)

    preds = model.predict(X_test)
    report = {}

    for attr, spec in cfg.get("fairness_attrs", {}).items():
        if attr not in raw_test_df.columns:
            continue
        if spec.get("bins"):
            groups = pd.cut(raw_test_df[attr], bins=spec["bins"], labels=spec["labels"])
        else:
            groups = raw_test_df[attr].astype(str)

        attr_report = {}
        for g in groups.dropna().unique():
            mask = (groups == g).values
            if mask.sum() < 5:
                continue  # too few samples to report reliably
            attr_report[str(g)] = {
                "n": int(mask.sum()),
                "accuracy": round(float(accuracy_score(y_test[mask], preds[mask])), 3),
                "recall": round(float(recall_score(y_test[mask], preds[mask], zero_division=0)), 3),
            }
        report[attr] = attr_report

    _plot_fairness(report, disease_key)
    return report


def _plot_fairness(report: dict, disease_key: str):
    if not report:
        return
    n_attrs = len(report)
    fig, axes = plt.subplots(1, n_attrs, figsize=(6 * n_attrs, 4), squeeze=False)
    for i, (attr, groups) in enumerate(report.items()):
        ax = axes[0][i]
        labels = list(groups.keys())
        accs = [groups[g]["accuracy"] for g in labels]
        ax.bar(labels, accs, color="#028090")
        ax.set_title(f"Accuracy by {attr} ({disease_key})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"fairness_{disease_key}.png"))
    plt.close()



if __name__ == "__main__":
    for disease in ["diabetes", "heart"]:
        print(f"\n=== {disease.upper()} ===")

        X_train, X_test, y_train, y_test, feature_names, raw_test = get_train_test_split(disease)
        sample = X_test[0:1]

        try:
            contributions, plot_path = explain_with_shap(sample, feature_names, disease, X_train)
            print("Top reasons:", get_top_reasons(contributions))
            domain_check = cross_check_with_domain_knowledge(contributions, disease)
            print("Domain cross-check:", domain_check)
        except ImportError:
            print("SHAP not installed -- skipping explanation demo (pip install shap)")

        fairness_report = evaluate_subgroup_fairness(disease)
        print("Fairness report:", fairness_report)
