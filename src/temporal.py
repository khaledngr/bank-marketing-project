"""Supplementary check - time-ordered split: does the model hold up on a later period?

The main evaluation uses a random stratified split, so the test clients come from the same
2008-2010 period as the training clients. The file is in date order and the yes-rate climbs
from 3% to 47% (EDA), so this check trains on the first 80% of rows and scores the last 20%.
It is reported next to the main test, never instead of it: the main test set is not touched.

Same recipe as the final model: tuned LightGBM settings from step 6, threshold chosen on
out-of-fold scores of the training rows only.

Run: python -m src.temporal
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import path
from src.data import LEAKY, load_frame
from src.evaluation import best_threshold, expected_calibration_error, out_of_fold, profit, summarize
from src.models import build_pipeline
from src.tracking import log_candidate, parent_run


def main() -> None:
    df = load_frame().drop(columns=LEAKY)          # file order = date order
    cut = int(0.8 * len(df))
    early, late = df.iloc[:cut], df.iloc[cut:]
    X_early, y_early = early.drop(columns="y"), early["y"]
    X_late, y_late = late.drop(columns="y"), late["y"]
    params = json.loads((path("tables") / "best_params.json").read_text())["lightgbm"]

    oof, _ = out_of_fold(build_pipeline("lightgbm", **params), X_early, y_early)
    threshold = best_threshold(y_early, oof)
    model = build_pipeline("lightgbm", **params).fit(X_early, y_early)
    scores = model.predict_proba(X_late)[:, 1]

    everyone = profit(y_late, np.ones(len(y_late), dtype=bool)) * 1000 / len(y_late)
    metrics = {**summarize(y_late, scores, threshold),
               "yes_rate_train": y_early.mean(), "yes_rate_test": y_late.mean(),
               "mean_predicted": scores.mean(), "brier": brier_score_loss(y_late, scores),
               "ECE": expected_calibration_error(y_late, scores),
               "profit_call_everyone": everyone}
    table_file = path("tables") / "temporal_check.csv"
    with parent_run("temporal_check", [table_file]):
        pd.DataFrame([metrics]).to_csv(table_file, index=False)
        log_candidate("first 80% rows -> last 20%", params, metrics)

    print(f"train rows 1-{cut} (yes {y_early.mean():.1%}) -> test rows {cut + 1}-{len(df)} "
          f"(yes {y_late.mean():.1%})")
    for key, value in metrics.items():
        print(f"  {key:22s} {value:.3f}")


if __name__ == "__main__":
    main()
