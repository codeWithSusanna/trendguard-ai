# Team Workflow — TrendGuard AI

4 people, 4 files, 1 shared repo. Read your section, do your part on
your own branch, open a Pull Request when done.

---

## Git Workflow (everyone follows this)

1. `main` branch is always the last known-working version. Nobody
   pushes to `main` directly.
2. Each person creates their OWN branch:
   ```bash
   git checkout -b feature/<your-part>
   # e.g. feature/trend-preprocessing, feature/model-training,
   #      feature/explainability-fairness, feature/frontend
   ```
3. Commit often with clear messages:
   ```bash
   git add src/preprocessing.py
   git commit -m "Add trend feature engineering for Gap 1"
   git push origin feature/trend-preprocessing
   ```
4. Open a Pull Request into `main` on GitHub. At least ONE other
   teammate reviews it before merging (catches bugs early).
5. Before you start work each day: `git pull origin main` so you have
   everyone else's latest changes.
6. If two people touch the same file and get a merge conflict — talk
   to each other before resolving it blindly.

**Suggested order:** Person A merges first (everyone depends on
preprocessing) → Person B next → Person C → Person D last (needs
everyone else's code to integrate).

---

## Person 1 — Data & Trend Engineer
**Owns:** `src/preprocessing.py`, `config/datasets.yaml`
**Implements:** Gap 1 (trend features), lays groundwork for Gap 7

### Your job
1. Get the real datasets from Kaggle (Pima Diabetes + Heart Disease
   UCI) into `data/diabetes.csv` and `data/heart.csv`.
2. Review `add_trend_features()` in `preprocessing.py` — it currently
   *simulates* 2 earlier checkups per patient since the public
   datasets are single-visit. Your real task: try to find or build
   an actual multi-visit version (even partially), OR clearly justify
   in the report why simulation was used instead.
3. Add/tune more trend features if useful (e.g. `Insulin_change`).
4. Make sure `get_train_test_split("diabetes")` and
   `get_train_test_split("heart")` both run cleanly.

### Deliverable
A `preprocessing.py` that any teammate can call like:
```python
from src.preprocessing import get_train_test_split
X_train, X_test, y_train, y_test, feature_names, raw_test_df = get_train_test_split("diabetes")
```

### Difficulty: Medium

---

## Person 2 — ML Engineer
**Owns:** `src/model_training.py`
**Implements:** Gap 7 (multi-disease pipeline)

### Your job
1. Run `python src/model_training.py` — it should train models for
   EVERY disease listed in `config/datasets.yaml` automatically.
2. Try adding one more model (e.g. SVM, Gradient Boosting) to
   `get_candidate_models()` and compare results.
3. Once Person 1 delivers real data, retrain and record final
   accuracy/F1/ROC-AUC for both diseases — this goes in the report.
4. If you add a 3rd disease later, you should NOT need to touch this
   file at all — only `config/datasets.yaml` changes. Test that
   claim; if it breaks, fix `model_training.py` so it holds.

### Deliverable
`models/best_model_diabetes.pkl`, `models/best_model_heart.pkl`, and
a metrics table (screenshot or CSV) for the report.

### Difficulty: Medium (Easy once Person 1's data is ready)

---

## Person 3 — Explainability & Fairness Lead
**Owns:** `src/explainability.py`
**Implements:** Gap 2 (domain cross-check) + Gap 6 (fairness audit)

### Your job — Gap 2
1. `pip install shap`, then check `explain_with_shap()` runs and
   produces a bar chart in `plots/`.
2. Look at `cross_check_with_domain_knowledge()` — it compares
   SHAP's top features against `domain_top_factors` in the config.
   Research (using medical sources, not guesses) what the REAL
   top clinical risk factors are for diabetes/heart disease and
   update the config if needed.
3. Write 2–3 sentences for the report on what it means when SHAP's
   top factor does NOT match the clinical list (see the disclaimer
   already in the code — expand on it).

### Your job — Gap 6
1. Run `evaluate_subgroup_fairness("diabetes")` and
   `evaluate_subgroup_fairness("heart")`.
2. Identify which subgroup (if any) has noticeably lower accuracy or
   recall. This is your key finding for the report.
3. Optionally: suggest ONE mitigation idea (e.g. rebalancing training
   data for the weak subgroup) — you don't have to implement it,
   just propose it.

### Deliverable
Console output / screenshots of the domain cross-check and fairness
report for both diseases, plus your written interpretation.

### Difficulty: Medium–Hard (SHAP internals can be fiddly; fairness logic is already built for you)

---

## Person 4 — Frontend/Backend + Integration Lead
**Owns:** `app.py`, and the GitHub repo itself (merging PRs, resolving conflicts)

### Your job
1. Once Person 1–3's branches are merged into `main`, pull latest and
   run `streamlit run app.py`.
2. Test the full flow: pick a disease → enter patient values → see
   prediction → see SHAP graph + domain cross-check → open the
   fairness expander and run the audit.
3. Polish the UI: better labels, grouping trend fields together,
   maybe a short "About this project" section explaining the 4 gaps
   to whoever is viewing the demo.
4. You are the final integration checkpoint — if any teammate's
   module doesn't plug in cleanly, work with them to fix the
   function signature/interface, not by rewriting their logic.

### Deliverable
A working `streamlit run app.py` that demonstrates all 4 gaps live,
plus you're responsible for the final repo being clean and
`README.md` being accurate before submission.

### Difficulty: Medium (mostly integration + UI, not new algorithms)

---

## Suggested Timeline (2-week sprint example)

| Days | What happens |
|---|---|
| 1–2 | Everyone reads this doc + their module; Person 1 starts data collection |
| 3–5 | Person 1 finishes preprocessing + trend features; Person 2 starts on top of it |
| 5–8 | Person 2 finishes training; Person 3 starts explainability + fairness |
| 8–11 | Person 3 finishes; Person 4 starts integration |
| 11–13 | Person 4 integrates everything, team tests together |
| 14 | Final polish, report writing, rehearsal for demo/viva |
