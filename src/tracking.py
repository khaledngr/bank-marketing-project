"""Step 7 - Experiment tracking with MLflow.

Every experiment script opens one parent run (the comparison as a whole: its table and figure)
and logs one child run per candidate (its parameters, CV metrics and per-fold AP). Optuna trials
are logged the same way, as children of their study. MLflow also records the git commit of
each run automatically, so any number can be traced back to the exact code that produced it.

Browse the runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   -> http://127.0.0.1:5000
"""
from contextlib import contextmanager
from pathlib import Path

import mlflow

from src.config import CONFIG, ROOT, SEED

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
