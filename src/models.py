"""Steps 4-5 - Three models, each wrapped in the same single Pipeline, compared on the same metrics.

    raw client data
      -> features   (rule-based new columns, see src/features.py)
      -> columns    (ColumnTransformer: impute + scale numbers, one-hot categories)
      -> model      (logistic regression | random forest | LightGBM)

The Pipeline is what gets cross-validated, tuned and deployed, so every fold refits the
preprocessing and the API later receives raw client fields, never a hand-made matrix.

Models use library defaults here on purpose: this is the untuned comparison; Optuna tunes them
in step 6. Every model is judged on out-of-fold scores, with its own profit-optimal threshold
(the remedy chosen in step 3).

Run: python -m src.models
"""
import time

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED, path
from src.data import load_split
from src.evaluation import best_threshold, out_of_fold, profit, summarize
from src.features import make_preprocessing
from src.plots import pr_and_profit
from src.tracking import log_candidate, parent_run

# Ordered from simplest to most flexible, so the table reads as "what each step bought".
MODELS = {
    "logistic_regression": lambda **p: LogisticRegression(max_iter=2000, random_state=SEED, **p),
    "random_forest": lambda **p: RandomForestClassifier(n_jobs=-1, random_state=SEED, **p),
    "lightgbm": lambda **p: LGBMClassifier(random_state=SEED, verbose=-1, **p),
}


def build_pipeline(name: str, **params) -> Pipeline:
    """The one Pipeline used everywhere: preprocessing + the chosen model."""
    return Pipeline([*make_preprocessing().steps, ("model", MODELS[name](**params))])


def main() -> None:
    X, _, y, _ = load_split()

    # Baseline a manager would propose without any model: call every client.
    call_everyone = pd.Series(True, index=y.index)
    rows = [{"model": "call everyone (no model)", "AP": y.mean(), "ROC_AUC": 0.5,
             "threshold": 0.0, "precision": y.mean(), "recall": 1.0, "share_called": 1.0,
             "profit_per_1000": profit(y, call_everyone) * 1000 / len(y)}]
    scores = {}
    table_file = path("tables") / "models_comparison.csv"
    with parent_run("models_comparison", [table_file, path("figures") / "models_pr_and_profit.png"]):
        for name in MODELS:
            pipe = build_pipeline(name)
            start = time.perf_counter()
            oof, fold_ap = out_of_fold(pipe, X, y)
            seconds = time.perf_counter() - start
            scores[name] = oof
            metrics = {**summarize(y, oof, best_threshold(y, oof)),
                       # CV convention (S5): AP is the mean of the 5 fold scores, with its spread.
                       "AP": fold_ap.mean(), "AP_fold_std": fold_ap.std(), "cv_seconds": seconds}
            rows.append({"model": name, **metrics})
            log_candidate(name, pipe.named_steps["model"].get_params(), metrics, fold_ap)

        table = pd.DataFrame(rows)
        table.to_csv(table_file, index=False)
        print(table.round(3).to_string(index=False))
        pr_and_profit(y, scores, "models_pr_and_profit")


if __name__ == "__main__":
    main()
