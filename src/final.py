"""Final step - fit the chosen model on all training data and score the untouched test set ONCE.

Every decision is frozen before this script runs, all of it chosen on cross-validation:
- model and hyperparameters: tuned LightGBM (step 6, reports/tables/best_params.json);
- no recalibration (step 9: the raw probabilities are already honest);
- decision threshold: the profit-optimal one from out-of-fold scores (step 6, 0.11);
- fairness mitigation: one threshold per age group (step 10, reports/tables/fairness_thresholds.json).
Nothing here is tuned on the test set; the test numbers only report how well those choices hold.

Outputs: the fitted model for the API (artifacts/models/final_model.joblib, also logged to MLflow),
test metrics, a test reliability diagram and the fairness comparison before/after mitigation.

Run: python -m src.final
"""
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import path
from src.data import load_split
from src.evaluation import expected_calibration_error, profit, summarize
from src.fairness import GROUPS, age_group, gaps, group_report, plot_groups
from src.models import build_pipeline
from src.plots import GREY, SERIES, save
from src.tracking import parent_run


def main() -> None:
    X_train, X_test, y_train, y_test = load_split()
    params = json.loads((path("tables") / "best_params.json").read_text())["lightgbm"]
    tuned = pd.read_csv(path("tables") / "tuning_results.csv")
    threshold = float(tuned.query("model == 'lightgbm' and setting == 'tuned'")["threshold"].iloc[0])
    group_thresholds = json.loads((path("tables") / "fairness_thresholds.json").read_text())["thresholds"]

    # Refit on the full 80% (S5: choose with CV, then refit on all development data).
    model = build_pipeline("lightgbm", **params).fit(X_train, y_train)
    bundle = {"model": model, "threshold": threshold, "group_thresholds": group_thresholds,
              "columns": list(X_train.columns)}
    model_file = path("models") / "final_model.joblib"
    joblib.dump(bundle, model_file)

    scores = model.predict_proba(X_test)[:, 1]
    groups = age_group(X_test["age"])
    cv = tuned.query("model == 'lightgbm' and setting == 'tuned'").iloc[0]

    # Headline metrics: test next to the cross-validation estimate, plus the no-model baseline.
    test = {**summarize(y_test, scores, threshold), "brier": brier_score_loss(y_test, scores),
            "ECE": expected_calibration_error(y_test, scores)}
    everyone = np.ones(len(y_test), dtype=bool)
    metrics = pd.DataFrame({
        "cross-validation (train)": {k: cv[k] for k in ["AP", "ROC_AUC", "threshold", "precision",
                                                        "recall", "share_called", "profit_per_1000"]},
        "test (final model)": test,
        "test: call everyone": {"AP": y_test.mean(), "ROC_AUC": 0.5, "threshold": 0.0,
                                "precision": y_test.mean(), "recall": 1.0, "share_called": 1.0,
                                "profit_per_1000": profit(y_test, everyone) * 1000 / len(y_test)},
    })

    # Fairness on the test set: the same frozen model, one threshold vs age-group thresholds.
    settings = {"before: one threshold": {g: threshold for g in GROUPS},
                "after: age-group thresholds": group_thresholds}
    reports, summary = [], []
    for name, thresholds in settings.items():
        report = group_report(y_test, scores, groups, thresholds).assign(setting=name)
        call = scores >= np.array([thresholds[g] for g in groups])
        summary.append({"setting": name, "profit_per_1000": profit(y_test, call) * 1000 / len(y_test),
                        **gaps(report)})
        reports.append(report)
    fairness = pd.concat(reports, ignore_index=True)
    fairness_summary = pd.DataFrame(summary)

    metrics_file = path("tables") / "final_test_metrics.csv"
    fairness_file = path("tables") / "final_test_fairness.csv"
    fairness_summary_file = path("tables") / "final_test_fairness_summary.csv"
    metrics.to_csv(metrics_file, index_label="metric")
    fairness.to_csv(fairness_file, index=False)
    fairness_summary.to_csv(fairness_summary_file, index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="perfect calibration")
    observed, predicted = calibration_curve(y_test, scores, n_bins=10, strategy="uniform")
    ax.plot(predicted, observed, "o-", color=SERIES[0], linewidth=2, markersize=6,
            label=f"final model (Brier {test['brier']:.3f}, ECE {test['ECE']:.3f})")
    ax.set(xlabel="Predicted probability", ylabel="Observed subscription rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Reliability on the untouched test set", loc="left")
    ax.legend(loc="upper left")
    save(fig, "final_test_reliability")
    plot_groups(fairness, "final_test_fairness")

    figures = [path("figures") / f"{f}.png" for f in ["final_test_reliability", "final_test_fairness"]]
    with parent_run("final_test", [metrics_file, fairness_file, fairness_summary_file, *figures]):
        mlflow.log_params({"model": "lightgbm", "threshold": threshold,
                           **{f"param_{k}": v for k, v in params.items()}})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test.items()})
        # MLflow saves models with skops, which only loads types we explicitly trust:
        # LightGBM, the column selector and our own feature functions.
        mlflow.sklearn.log_model(model, name="final_model", skops_trusted_types=[
            "collections.OrderedDict", "numpy.dtype", "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier", "sklearn.compose._column_transformer.make_column_selector",
            "src.features.add_features", "src.features.contacted_before",
            "src.features.life_stage", "src.features.log_skewed"])

    print(metrics.round(4).to_string())
    print()
    cols = ["setting", "group", "threshold", "clients", "subscribers", "call_rate", "recall",
            "recall_ci_low", "recall_ci_high", "false_positive_rate", "precision", "ECE"]
    print(fairness[cols].round(3).to_string(index=False))
    print()
    print(fairness_summary.round(3).to_string(index=False))
    print(f"\nModel saved to {model_file}")


if __name__ == "__main__":
    main()
