"""
MODULE 4 (Person D): FRONTEND + BACKEND INTEGRATION
======================================================
Owns: the Streamlit app (acts as both UI and backend here), disease
selector (Gap 7), trend input fields (Gap 1), SHAP + domain
cross-check display (Gap 2), and the fairness report tab (Gap 6).

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `import src.x`

from src.preprocessing import preprocess_single_patient, load_config
from src.model_training import predict_risk, load_model
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="TrendGuard AI", layout="centered")

with open(os.path.join(ROOT, "config", "datasets.yaml")) as f:
    ALL_DISEASES = yaml.safe_load(f)

st.title("🩺 TrendGuard AI")
st.caption("Explainable, trend-aware, fairness-checked disease risk prediction")

# ---- GAP 7: disease selector ----
disease_key = st.selectbox(
    "Select disease model",
    options=list(ALL_DISEASES.keys()),
    format_func=lambda k: ALL_DISEASES[k]["display_name"],
)

try:
    model, feature_names = load_model(disease_key)
except FileNotFoundError:
    st.error(f"No trained model for '{disease_key}'. Run `python src/model_training.py` first.")
    st.stop()

st.subheader("Patient Details")
st.caption("Fields ending in *_change / *_trend_slope capture Gap 1 (trend over checkups).")

col1, col2 = st.columns(2)
patient_input = {}
for i, feat in enumerate(feature_names):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        patient_input[feat] = st.number_input(feat, value=0.0, format="%.2f")

if st.button("Predict Risk", type="primary"):
    scaled_row = preprocess_single_patient(patient_input, feature_names, disease_key)
    result = predict_risk(scaled_row, disease_key)

    st.subheader("Prediction Result")
    if result["prediction"] == 1:
        st.error(f"⚠️ High Risk — {result['risk_percent']}%")
    else:
        st.success(f"✅ Low Risk — {result['risk_percent']}%")

    # ---- GAP 2: SHAP + domain cross-check ----
    st.subheader("Why this result?")
    try:
        from src.explainability import explain_with_shap, get_top_reasons, cross_check_with_domain_knowledge
        contributions, plot_path = explain_with_shap(scaled_row, feature_names, disease_key)
        st.write(get_top_reasons(contributions))
        if os.path.exists(plot_path):
            st.image(plot_path)

        domain_check = cross_check_with_domain_knowledge(contributions, disease_key)
        st.info(
            f"**Domain cross-check:** {domain_check['agreement_percent']}% of the top factors "
            f"match known clinical risk factors for this disease "
            f"({', '.join(domain_check['overlap']) or 'none'}).\n\n"
            f"⚠️ {domain_check['disclaimer']}"
        )
    except ImportError:
        st.warning("Install SHAP to see explanations: `pip install shap`")

# ---- GAP 6: fairness tab ----
with st.expander("📊 Model Fairness Report (Gap 6)"):
    st.caption("Accuracy/recall broken down by subgroup — flags if the model is weaker for any group.")
    if st.button("Run fairness audit"):
        from src.explainability import evaluate_subgroup_fairness
        report = evaluate_subgroup_fairness(disease_key)
        for attr, groups in report.items():
            st.write(f"**By {attr}:**")
            st.table(groups)
        fairness_plot = os.path.join(ROOT, "plots", f"fairness_{disease_key}.png")
        if os.path.exists(fairness_plot):
            st.image(fairness_plot)

st.markdown("---")
st.caption("Educational project demo — not a substitute for professional medical advice.")
