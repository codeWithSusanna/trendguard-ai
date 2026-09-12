# Contributing to TrendGuard AI

## First-time setup (each person, once)

```bash
git clone <repo-url>
cd trendguard-ai
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Daily workflow

```bash
git checkout main
git pull origin main
git checkout -b feature/<your-part>   # only if starting new work
# ... make changes ...
git add <files>
git commit -m "Clear message about what you did"
git push origin feature/<your-part>
```

Then open a Pull Request on GitHub into `main`.

## Branch names to use

| Person | Branch |
|---|---|
| Data & Trend Engineer | `feature/trend-preprocessing` |
| ML Engineer | `feature/model-training` |
| Explainability & Fairness Lead | `feature/explainability-fairness` |
| Frontend/Backend Lead | `feature/frontend-integration` |

## Before opening a PR

- [ ] Code runs without errors: `python src/<your_file>.py`
- [ ] You didn't commit `models/*.pkl` or `plots/*.png` (they're
      gitignored — auto-generated, shouldn't be in git)
- [ ] You pulled latest `main` and resolved any conflicts
- [ ] One teammate has reviewed your PR

## Commit message style

Keep it short and specific:
- ✅ `Add glucose trend slope feature`
- ✅ `Fix fairness report crashing on small subgroups`
- ❌ `updates`
- ❌ `fix stuff`
