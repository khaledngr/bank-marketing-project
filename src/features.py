"""Step 2 - Feature engineering and preprocessing.

Two layers, both meant to sit inside the model Pipeline:
1. `add_features`: stateless row-by-row rules (no statistics learned), so it cannot leak.
2. `make_preprocessing`: imputer, scaler and one-hot encoder, which DO learn from data and are
   therefore refitted inside every CV fold by the Pipeline.

Run: python -m src.features   (feature ablation with 5-fold CV on the training set)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import SEED, path
from src.data import cv_splitter, load_split
from src.tracking import log_candidate, parent_run


def contacted_before(X: pd.DataFrame) -> pd.DataFrame:
    """pdays = -1 is a code for "never contacted", not a number of days. Left as -1, a linear
    model would read it as "contacted yesterday". We flag it explicitly and turn the code into
    a missing value, which the median imputer fills inside each fold."""
    X["contacted_before"] = (X["pdays"] != -1).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != -1)
    return X


def log_skewed(X: pd.DataFrame) -> pd.DataFrame:
    """balance, campaign and previous have long right tails (balance up to 102k EUR, up to 63 calls).
    Logs stop a few extreme clients from dominating the logistic-regression coefficients.
    Balance can be negative (overdraft), hence the signed log."""
    X["balance"] = np.sign(X["balance"]) * np.log1p(X["balance"].abs())
    X["campaign"] = np.log1p(X["campaign"])
    X["previous"] = np.log1p(X["previous"])
    return X


def life_stage(X: pd.DataFrame) -> pd.DataFrame:
    """EDA showed a U-shape: under-25s (26%) and over-60s (34%) subscribe far more than the
    30-59 middle (~10%), likely students and retirees with free savings. A straight-line model
    cannot draw a U from raw age, so we give it the three stages as a category.
    Note: derived from age, our sensitive attribute; the fairness audit must account for it."""
    X["life_stage"] = np.select([X["age"] < 25, X["age"] >= 60], ["young", "senior"], "working")
    return X


FEATURE_STEPS = [contacted_before, log_skewed, life_stage]


def add_features(X: pd.DataFrame, steps=tuple(FEATURE_STEPS)) -> pd.DataFrame:
    X = X.copy()
    for step in steps:
        X = step(X)
    return X


def make_preprocessing(steps=tuple(FEATURE_STEPS)) -> Pipeline:
    """Feature rules, then numbers -> impute + scale, categories -> one-hot.

    'unknown' (job, education, contact, poutcome) is kept as its own category on purpose: it is
    not random noise (contact = 'unknown' converts at 4% vs 15% for cellular), so imputing the
    most frequent value would erase real signal. handle_unknown='ignore' lets the deployed model
    accept a category it never saw in training instead of crashing.
    """
    columns = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]),
         make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features, kw_args={"steps": tuple(steps)})),
        ("columns", columns),
    ])


def main() -> None:
    """Add the feature steps one at a time and measure what each one buys, per model family."""
    X, _, y, _ = load_split()
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "lightgbm": LGBMClassifier(random_state=SEED, verbose=-1),
    }
    rows = []
    table_file = path("tables") / "features_ablation.csv"
    with parent_run("feature_ablation", [table_file]):
        for k in range(len(FEATURE_STEPS) + 1):
            label = "raw columns" if k == 0 else "+ " + FEATURE_STEPS[k - 1].__name__
            for name, model in models.items():
                pipe = Pipeline([("prep", make_preprocessing(FEATURE_STEPS[:k])), ("model", model)])
                cv = cross_validate(pipe, X, y, cv=cv_splitter(),
                                    scoring=["average_precision", "roc_auc"])
                # Same metric names as every other experiment, so MLflow lines them up.
                metrics = {
                    "AP": cv["test_average_precision"].mean(),
                    "AP_fold_std": cv["test_average_precision"].std(),
                    "ROC_AUC": cv["test_roc_auc"].mean(),
                }
                rows.append({"features": label, "model": name, **metrics})
                log_candidate(f"{name} | {label}", {"model": name, "features": label},
                              metrics, cv["test_average_precision"])

        table = pd.DataFrame(rows)
        table.to_csv(table_file, index=False)
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
