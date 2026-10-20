"""Supplementary check - nested cross-validation of the tuned LightGBM.

The reported CV score (AP 0.459) is slightly optimistic: the same 5 folds chose the
hyperparameters, scored them and picked the threshold (S5). Nested CV removes that bias:
- outer loop: the usual 5 stratified folds of the training set;
- inner loop: on the 4 outer-training folds only, Optuna searches the same space (fewer trials,
  `tuning.nested_trials`) with its own 5-fold CV, and the profit threshold is chosen on the
  inner out-of-fold scores;
- the winner is refitted on the 4 folds and scored once on the outer fold it never saw.
The test set is not touched.

Run: python -m src.nested_cv      (about 15 minutes)
"""
import optuna
import pandas as pd

from src.config import CONFIG, SEED, path
from src.data import cv_splitter, load_split
from src.evaluation import best_threshold, out_of_fold, summarize
from src.models import build_pipeline
from src.tracking import log_candidate, parent_run
from src.tuning import sample

LGBM = CONFIG["tuning"]["lightgbm"]


def tune_inner(X: pd.DataFrame, y: pd.Series) -> dict:
    """Optuna on the outer-training folds only; returns the best settings."""
    def objective(trial: optuna.Trial) -> float:
        params = {**LGBM["fixed"], **sample(trial, LGBM["space"])}
        _, fold_ap = out_of_fold(build_pipeline("lightgbm", **params), X, y)
        return fold_ap.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=CONFIG["tuning"]["nested_trials"])
    return {**LGBM["fixed"], **study.best_params}


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, _, y, _ = load_split()

    rows = []
    table_file = path("tables") / "nested_cv.csv"
    with parent_run("nested_cv", [table_file]):
        for fold, (train_idx, val_idx) in enumerate(cv_splitter().split(X, y), start=1):
            X_in, y_in = X.iloc[train_idx], y.iloc[train_idx]
            params = tune_inner(X_in, y_in)
            inner_oof, _ = out_of_fold(build_pipeline("lightgbm", **params), X_in, y_in)
            threshold = best_threshold(y_in, inner_oof)

            model = build_pipeline("lightgbm", **params).fit(X_in, y_in)
            scores = model.predict_proba(X.iloc[val_idx])[:, 1]
            metrics = summarize(y.iloc[val_idx], scores, threshold)
            rows.append({"outer_fold": fold, **metrics})
            log_candidate(f"nested outer fold {fold}", params, metrics)
            print(f"outer fold {fold}: AP {metrics['AP']:.3f}, threshold {threshold:.2f}, "
                  f"profit {metrics['profit_per_1000']:.0f} per 1,000")

        table = pd.DataFrame(rows)
        table.to_csv(table_file, index=False)
    print(f"\nnested CV: AP {table['AP'].mean():.3f} (fold std {table['AP'].std(ddof=0):.3f}), "
          f"profit {table['profit_per_1000'].mean():.0f} per 1,000")


if __name__ == "__main__":
    main()
