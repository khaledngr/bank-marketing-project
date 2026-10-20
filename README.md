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
python3.13 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.eda
```

The dataset downloads automatically on the first run. No API keys needed.

## Pipeline

Run each step from the project root, with the virtual environment active, in this order
(steps 8-11 read the settings chosen in steps 6 and 10).

| # | Step | Command | Status |
|---|------|---------|--------|
| 1 | EDA, problem framing, stratified split | `python -m src.eda` | done |
| 2 | Feature engineering + ablation | `python -m src.features` | done |
| 3 | Class imbalance: remedies + cost-based threshold | `python -m src.imbalance` | done |
| 4 | Model comparison: logistic regression, random forest, LightGBM | `python -m src.models` | done |
| 5 | One Pipeline: features + ColumnTransformer + model | `build_pipeline()` in `src/models.py` | done |
| 6 | Optuna tuning with stratified CV (30 trials per model) | `python -m src.tuning` | done |
| 7 | Experiment tracking (MLflow) | logged automatically by steps 2-4, 6, 8-11 and the supplementary checks | done |
| 8 | Explainability: gain, permutation, SHAP (global + 4 clients) | `python -m src.explain` | done |
| 9 | Calibration: reliability diagram, Brier, ECE; raw vs Platt vs isotonic | `python -m src.calibration` | done |
| 10 | Fairness audit on age groups + mitigation (designed on CV; test comparison in the final step) | `python -m src.fairness` | done |
| 11 | Final model + one-time test evaluation (all decisions frozen) | `python -m src.final` | done |
| 12 | Prediction API (FastAPI) | `uvicorn src.api:app --reload` | done |
| + | Supplementary: time-ordered split (first 80% of rows -> last 20%) | `python -m src.temporal` | done |
| + | Supplementary: nested CV of the tuned LightGBM (about 15 min) | `python -m src.nested_cv` | done |

### Serving predictions

After step 11 has saved the model, start the API:

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs, choose **POST /predict**, then **Try it out** and **Execute**.
Or from a terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"age": 67, "job": "retired", "marital": "married", "education": "tertiary", "default": "no", "balance": 4200, "housing": "no", "loan": "no", "contact": "cellular", "day": 12, "month": "oct", "campaign": 1, "pdays": 95, "previous": 2, "poutcome": "success"}'
```

The answer gives the probability of subscribing, the expected value of the call in EUR, and two
decisions: `call` (one profit-optimal threshold) and `call_age_fair` (age-group thresholds from
the fairness step). Invalid inputs (unknown category, age under 18, ...) are rejected with a
message naming the field. `duration` is not an input: it is only known after the call.

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
  tracking.py     step 7: MLflow logging (one parent run per experiment, one child per candidate);
                  `python -m src.tracking` plots every logged Optuna trial
  explain.py      step 8: global importance (3 views) + local SHAP explanations
  calibration.py  step 9: are the probabilities honest? (reliability, Brier, ECE)
  fairness.py     step 10: age-group audit (equal opportunity) + age-group thresholds
  final.py        step 11: refit on all training data, score the test set once, save the model
  temporal.py     supplementary: train on the earliest 80% of rows, score the latest 20% (drift)
  nested_cv.py    supplementary: nested CV, so tuning does not score its own choices
  api.py          FastAPI app: POST /predict for one client, GET /health
  evaluation.py   shared metrics: out-of-fold scores, AP, precision, recall, profit, ECE
  plots.py        shared figure style
reports/
  figures/        plots used in the report
  tables/         CSV results used in the report
report/
  main.tex        the written report (LaTeX); build with `tectonic -X compile report/main.tex`
  main.pdf        the compiled report
artifacts/models/ final_model.joblib, used by the API (git-ignored, rebuilt by step 11)
mlflow.db, mlruns/ MLflow runs and their artifacts (git-ignored, rebuilt by re-running the steps)
data/             dataset cache (git-ignored)
```

## Results on the untouched test set

Tuned LightGBM, threshold 0.11 chosen on cross-validation, scored once on the 20% test set
(9,043 clients).

| | Test | Cross-validation | Call everyone |
|---|---|---|---|
| Average precision | 0.476 | 0.459 | 0.117 |
| ROC-AUC | 0.809 | 0.802 | 0.5 |
| Clients called | 24% | 24% | 100% |
| Subscribers reached (recall) | 67% | 66% | 100% |
| Calls that convert (precision) | 33% | 32% | 12% |
| Profit per 1,000 clients | EUR 2,711 | EUR 2,676 | EUR 850 |
| Calibration (Brier / ECE) | 0.080 / 0.010 | 0.081 / 0.006 | |

Fairness (equal opportunity across age groups, test set): the recall gap between age groups
drops from 35 to 5 points with age-group thresholds, at a cost of 6% of profit
(EUR 2,711 to EUR 2,551 per 1,000 clients). Details in `reports/tables/final_test_*.csv`.

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
