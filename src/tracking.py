"""Step 7 - Experiment tracking with MLflow.

Every experiment script opens one parent run (the comparison as a whole: its table and figure)
and logs one child run per candidate (its parameters, CV metrics and per-fold AP). Optuna trials
are logged the same way, as children of their study. MLflow also records the git commit of
each run automatically, so any number can be traced back to the exact code that produced it.

Browse the runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   -> http://127.0.0.1:5000
Plot the logged Optuna trials (after step 6):  python -m src.tracking
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from src.config import CONFIG, ROOT, SEED
from src.plots import SERIES, save

TRACKING = CONFIG["tracking"]


def setup() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / TRACKING['db']}")
    if mlflow.get_experiment_by_name(TRACKING["experiment"]) is None:
        mlflow.create_experiment(
            TRACKING["experiment"], artifact_location=(ROOT / TRACKING["artifacts"]).as_uri()
        )
    mlflow.set_experiment(TRACKING["experiment"])


@contextmanager
def parent_run(name: str, artifacts: list[Path] = ()):
    """One run per experiment script; the artifacts are logged once the body has written them."""
    setup()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"seed": SEED, **CONFIG["data"], **CONFIG["business"]})
        yield run
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))


def log_candidate(name: str, params: dict, metrics: dict, fold_ap=()) -> None:
    """One child run: what was tried, and how it scored."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for fold, ap in enumerate(fold_ap, start=1):
            mlflow.log_metric("fold_AP", float(ap), step=fold)


def main() -> None:
    """Read every Optuna trial back from the MLflow store and plot it: proof the tuning was
    logged, and a view of how stable each model's score is across settings."""
    setup()
    runs = mlflow.search_runs(experiment_names=[TRACKING["experiment"]])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for color, model in zip(SERIES, ["logistic_regression", "random_forest", "lightgbm"]):
        trials = runs[runs["tags.mlflow.runName"].str.startswith(f"{model} trial")].copy()
        trials["trial"] = trials["tags.mlflow.runName"].str.split().str[-1].astype(int)
        trials = trials.sort_values("trial")
        ax.errorbar(trials["trial"], trials["metrics.AP"], yerr=trials["metrics.AP_fold_std"],
                    fmt="o", color=color, markersize=4, elinewidth=0.8, alpha=0.85,
                    label=f"{model}: {len(trials)} trials, best AP {trials['metrics.AP'].max():.3f}")
    ax.set(xlabel="Optuna trial", ylabel="Mean CV AP (bar = fold std)")
    ax.set_title("Every Optuna trial, read back from the MLflow store", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "mlflow_trials")


if __name__ == "__main__":
    main()
