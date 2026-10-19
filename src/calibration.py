"""Step 9 - Calibration: when the model says 20%, do 20% of those clients subscribe?

Our call rule (call if P(subscribe) > 0.10) only makes sense if the probabilities are honest, so
we check them before trusting the rule:
- reliability diagram: mean predicted vs observed rate per bin, with the number of clients
  behind each point (small bins are noisy);
- Brier score: overall squared error of the probabilities (mixes calibration and ranking);
- ECE: average gap per bin, weighted by bin size.

Three candidates on the tuned LightGBM, all scored out-of-fold (S7):
- uncalibrated;
- Platt scaling (sigmoid) and isotonic regression via CalibratedClassifierCV. Inside each outer
  training fold, the calibrator is fitted on inner-fold predictions the model never trained on,
  so no calibrator ever sees its own training labels.
S7 rule: prefer the simpler option when the differences are small.

Run: python -m src.calibration     (a few minutes)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import path
from src.data import cv_splitter, load_split
from src.evaluation import best_threshold, expected_calibration_error, out_of_fold, summarize
from src.models import build_pipeline
from src.plots import GREY, SERIES, save
from src.tracking import log_candidate, parent_run

N_BINS = 10


def candidates(params: dict) -> dict:
    base = build_pipeline("lightgbm", **params)
    return {
        "uncalibrated": base,
        "Platt (sigmoid)": CalibratedClassifierCV(base, method="sigmoid", cv=cv_splitter()),
        "isotonic": CalibratedClassifierCV(base, method="isotonic", cv=cv_splitter()),
    }


def main() -> None:
    X, _, y, _ = load_split()
    params = json.loads((path("tables") / "best_params.json").read_text())["lightgbm"]

    rows, scores = [], {}
    table_file = path("tables") / "calibration.csv"
    bins_file = path("tables") / "calibration_bins.csv"
    figure = path("figures") / "calibration_reliability.png"
    with parent_run("calibration", [table_file, bins_file, figure]):
        for name, model in candidates(params).items():
            oof, fold_ap = out_of_fold(model, X, y)
            scores[name] = oof
            metrics = {**summarize(y, oof, best_threshold(y, oof)),
                       "AP": fold_ap.mean(), "AP_fold_std": fold_ap.std(),
                       "brier": brier_score_loss(y, oof),
                       "ECE": expected_calibration_error(y, oof, N_BINS),
                       "mean_predicted": oof.mean(), "observed_rate": y.mean()}
            rows.append({"candidate": name, **metrics})
            log_candidate(f"calibration | {name}", {"method": name, **params}, metrics, fold_ap)

        table = pd.DataFrame(rows)
        table.to_csv(table_file, index=False)

        # Per-bin view of the uncalibrated model, with counts (S7: check the evidence per point).
        raw = scores["uncalibrated"]
        bin_id = np.minimum((raw * N_BINS).astype(int), N_BINS - 1)
        bins = (pd.DataFrame({"bin": bin_id, "p": raw, "y": y.to_numpy()})
                .groupby("bin").agg(clients=("y", "size"), mean_predicted=("p", "mean"),
                                     observed_rate=("y", "mean")))
        bins.index = [f"{b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}" for b in bins.index]
        bins["gap"] = bins["observed_rate"] - bins["mean_predicted"]
        bins.to_csv(bins_file, index_label="predicted_range")

        fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 7), height_ratios=[3, 1], sharex=True)
        ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="perfect calibration")
        for color, (name, oof) in zip(SERIES, scores.items()):
            observed, predicted = calibration_curve(y, oof, n_bins=N_BINS, strategy="uniform")
            ax.plot(predicted, observed, "o-", color=color, linewidth=2, markersize=6, label=name)
        ax.set(ylabel="Observed subscription rate", xlim=(0, 1), ylim=(0, 1))
        ax.set_title("Reliability diagram, out-of-fold (below diagonal = overestimates)", loc="left")
        ax.legend(loc="upper left")
        ax_hist.hist(raw, bins=np.linspace(0, 1, N_BINS + 1), color=SERIES[0], rwidth=0.9)
        ax_hist.set_yscale("log")
        ax_hist.set(xlabel="Predicted probability", ylabel="Clients (log scale)")
        ax_hist.set_title("How many clients sit behind each point (uncalibrated)", loc="left")
        save(fig, "calibration_reliability")

    print(table[["candidate", "AP", "brier", "ECE", "mean_predicted", "observed_rate",
                 "threshold", "profit_per_1000"]].round(4).to_string(index=False))
    print("\nUncalibrated model, per bin:")
    print(bins.round(3).to_string())


if __name__ == "__main__":
    main()
