# TrendGuard AI
**Explainable, Trend-Aware, Fairness-Checked Disease Risk Prediction**

This repo implements the base project **plus 4 research gaps**:

| Gap | What it means | Where it lives |
|---|---|---|
| **Gap 1** — Trend features | Uses change-over-time (not just one snapshot) | `src/preprocessing.py` |
| **Gap 2** — Domain cross-check | SHAP explanations checked against known clinical risk factors | `src/explainability.py` |
| **Gap 6** — Fairness audit | Accuracy/recall reported per age/gender subgroup | `src/explainability.py` |
| **Gap 7** — Multi-disease | Same pipeline runs for diabetes AND heart disease | `config/datasets.yaml` |

## Folder Structure

```
trendguard-ai/
├── config/
│   └── datasets.yaml         <- add a new disease here, nothing else
├── data/
│   ├── diabetes.csv          <- put real Kaggle CSV here
│   └── heart.csv             <- put real Kaggle CSV here
├── src/
│   ├── preprocessing.py      <- Person A
│   ├── model_training.py     <- Person B
│   └── explainability.py     <- Person C (Gap 2 + Gap 6)
├── app.py                    <- Person D (frontend + backend)
├── models/                   <- auto-generated .pkl files (gitignored)
├── plots/                    <- auto-generated charts (gitignored)
├── requirements.txt
└── docs/workflow.md          <- detailed step-by-step for each person
```

## Setup

```bash
git clone <your-repo-url>
cd trendguard-ai
pip install -r requirements.txt
```

Put the real datasets in `data/diabetes.csv` and `data/heart.csv`
(Kaggle: "Pima Indians Diabetes" and "Heart Disease UCI"). Without
them, the pipeline auto-generates synthetic data so everyone can
develop and test before the real CSVs are ready.

## Run Order

```bash
python src/model_training.py     # trains + saves models for BOTH diseases
python src/explainability.py     # runs SHAP + domain check + fairness audit (console demo)
streamlit run app.py             # full web app
```

## Team: 4 People, 4 Owned Files

See **`docs/workflow.md`** for the full breakdown, responsibilities,
daily flow, and Git branching model. Quick summary:

| # | Person | Owns | Gap(s) |
|---|--------|------|--------|
| 1 | Data & Trend Engineer | `src/preprocessing.py`, `config/datasets.yaml` | Gap 1, supports Gap 7 |
| 2 | ML Engineer | `src/model_training.py` | Gap 7 |
| 3 | Explainability & Fairness Lead | `src/explainability.py` | Gap 2, Gap 6 |
| 4 | Frontend/Backend + Integration Lead | `app.py`, repo/PR management | Connects everything |

## Data Flow

```
config/datasets.yaml (disease definitions)
        ↓
preprocessing.py  → clean data → trend features (Gap 1) → scaled train/test
        ↓
model_training.py → trains per disease (Gap 7) → best_model_<disease>.pkl
        ↓
explainability.py → SHAP → domain cross-check (Gap 2) → fairness audit (Gap 6)
        ↓
app.py → disease selector → patient form → prediction + explanation + fairness tab
```

## Note on the "Trend" data (Gap 1)

The public Kaggle datasets are single-visit (one row per patient), so
there is no real multi-checkup history to compute trends from. This
repo ships a clearly-logged **simulation mode** that fabricates two
earlier checkups per patient so the trend-feature *pipeline* can be
built and demoed end-to-end now. If your team can find or construct
a real multi-visit dataset (or generate one with a supervisor's
guidance), swap it in — `preprocessing.py` already supports real
longitudinal data via `patient_id_col` / `visit_col` in the config
with no other code changes needed.

This project is for academic/demo purposes only and is not a
substitute for professional medical advice.
