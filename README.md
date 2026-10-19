# Bank Marketing: who should the bank call?

MSc group project, *Supervised Learning & Model Engineering*, Albert School, October 2026.\
Team: Khaled · Alzira

End-to-end machine-learning proof of concept: from raw data to a served prediction endpoint.
A bank phones clients to sell a term deposit. Each call costs money and each subscription earns
money, so we predict who is likely to subscribe and call only the clients worth calling.

## At a glance

| | |
|---|---|
| **Data** | UCI Bank Marketing ([OpenML 1461](https://www.openml.org/d/1461)): 45,211 clients, 16 features |
| **Target** | `y = 1` if the client subscribed (11.7% of clients) |
| **Removed** | `duration`: only known after the call, so it would leak the answer |
| **Split** | 80% train (5-fold stratified CV for validation), 20% test scored once at the end |
| **Decision rule** | Call if P(subscribe) x EUR 50 > EUR 5 call cost, i.e. P > 0.10 (see `config.yaml`) |
| **Models** | Logistic regression, Random Forest, LightGBM, in one scikit-learn Pipeline |

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.eda
```

The dataset downloads automatically on the first run. No API keys needed.

## Pipeline

Run each step from the project root, with the virtual environment active, in this order
(steps 8-10 read the settings chosen in step 6).

| # | Step | Command | Status |
|---|------|---------|--------|
| 1 | EDA, problem framing, stratified split | `python -m src.eda` | done |
| 2 | Feature engineering + ablation | `python -m src.features` | done |
| 3 | Class imbalance: remedies + cost-based threshold | `python -m src.imbalance` | done |
| 4 | Model comparison: logistic regression, random forest, LightGBM | `python -m src.models` | done |
| 5 | One Pipeline: features + ColumnTransformer + model | `build_pipeline()` in `src/models.py` | done |
| 6 | Optuna tuning with stratified CV (30 trials per model) | `python -m src.tuning` | done |
| 7 | Experiment tracking (MLflow) | logged automatically by steps 2-4, 6, 8, 9 and 10 | done |
| 8 | Explainability: gain, permutation, SHAP (global + 4 clients) | `python -m src.explain` | done |
| 9 | Calibration: reliability diagram, Brier, ECE; raw vs Platt vs isotonic | `python -m src.calibration` | done |
| 10 | Fairness audit on age groups + mitigation (designed on CV; test comparison in the final step) | `python -m src.fairness` | done |
| | Prediction API (FastAPI) | | planned |

### Experiment tracking

Every experiment script logs to a local MLflow store: one parent run per script (its table and
figure as artifacts) and one child run per candidate (parameters, CV metrics, AP of each fold,
git commit). To browse them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open http://127.0.0.1:5000, switch to **Model training** and open the `bank-marketing`
experiment. On a Mac, port 5000 is often taken by AirPlay: add `--port 5001` and open that port instead.

## Project structure

```
config.yaml       paths, seed, split sizes, business costs, Optuna search spaces
src/
  config.py       loads config.yaml
  data.py         load, rename, target, drop leakage, train/test split, CV folds
  eda.py          step 1: figures and tables
  features.py     step 2: engineered features + preprocessing (imputer, scaler, one-hot)
  imbalance.py    step 3: compare remedies, pick the profit-maximising threshold
  models.py       steps 4-5: the single Pipeline + 3-model comparison
  tuning.py       step 6: Optuna search per model, search spaces in config.yaml
  tracking.py     step 7: MLflow logging (one parent run per experiment, one child per candidate)
  explain.py      step 8: global importance (3 views) + local SHAP explanations
  calibration.py  step 9: are the probabilities honest? (reliability, Brier, ECE)
  fairness.py     step 10: age-group audit (equal opportunity) + age-group thresholds
  evaluation.py   shared metrics: out-of-fold scores, AP, precision, recall, profit, ECE
  plots.py        shared figure style
reports/
  figures/        plots used in the report
  tables/         CSV results used in the report
artifacts/        fitted models (git-ignored)
mlflow.db, mlruns/ MLflow runs and their artifacts (git-ignored, rebuilt by re-running the steps)
data/             dataset cache (git-ignored)
```

## Guarantees against leakage

- `duration` is dropped before any split.
- The test set is split off once (seed 42) and only scored at the very end.
- All preprocessing sits inside the Pipeline, so it is refitted inside every CV fold.
- Class weights, SMOTE and calibrators are fitted inside training folds only; the decision
  threshold is chosen on out-of-fold scores, never on the test set.
- EDA findings come from the training set only (the test set appears only in the
  stratification check and the drift plot).
- One seed in `config.yaml` drives every split, model and sampler.

## Known limitation

UCI documents the file as ordered by date (campaigns from May 2008 to Nov 2010, no year column),
and the subscription rate rises from 3% to 47% across the file. A random split mixes periods,
so test scores are likely optimistic for future campaigns.
