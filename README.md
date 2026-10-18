# Bank Marketing: who should the bank call?

End-to-end machine-learning proof of concept: from raw data to a served prediction endpoint.
A bank phones clients to sell a term deposit. Each call costs money and each subscription earns
money, so we predict who is likely to subscribe and call only the clients worth calling.

MSc group project, *Supervised Learning & Model Engineering*, Albert School, October 2026.

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

The dataset downloads automatically on the first run. No API keys are needed
([`.env.example`](.env.example) only matters for a remote MLflow server).

## Pipeline

Run each step from the project root, with the virtual environment active.

| # | Step | Command | Status |
|---|------|---------|--------|
| 1 | EDA, problem framing, stratified split | `python -m src.eda` | done |
| 2 | Feature engineering | | planned |
| 3 | Class imbalance: metrics + remedy | | planned |
| 4 | Model comparison (3 models) | | planned |
| 5 | One Pipeline: ColumnTransformer + model | | planned |
| 6 | Optuna tuning with stratified CV | | planned |
| 7 | Experiment tracking (MLflow) | | planned |
| 8 | Explainability (SHAP) | | planned |
| 9 | Calibration | | planned |
| 10 | Fairness audit on age groups | | planned |
| | Prediction API (FastAPI) | | planned |

## Project structure

```
config.yaml       paths, seed, split sizes, business costs
src/
  config.py       loads config.yaml
  data.py         load, rename, target, drop leakage, train/test split, CV folds
  eda.py          step 1: figures and tables
reports/
  figures/        plots used in the report
  tables/         CSV results used in the report
artifacts/        fitted models (git-ignored)
data/             dataset cache (git-ignored)
```

## Guarantees against leakage

- `duration` is dropped before any split.
- The test set is split off once (seed 42) and only scored at the very end.
- All preprocessing sits inside the Pipeline, so it is refitted inside every CV fold.
- EDA uses the training set only.
- One seed in `config.yaml` drives every split, model and sampler.

## Known limitation

The rows are in time order (May 2008 to Nov 2010, no year column) and the subscription rate rises
from 3% to 47% across the file. A random split mixes periods, so test scores are likely
optimistic for future campaigns.

## Team

Khaled · Alzira
