"""Step 6 - Hyperparameter tuning with Optuna and stratified cross-validation.

For each of the three models, Optuna proposes hyperparameters (TPE sampler: it learns from past
trials where to look next), the whole Pipeline is scored by mean AP on the same 5 stratified
folds, and every trial is logged to MLflow as a child of that model's study run.

AP is the objective because it is threshold-free and focused on the rare class (step 3); the
profit-optimal threshold is chosen afterwards on the best model's out-of-fold scores.

The best CV score is slightly optimistic (the same folds picked the winner, see S5 nested CV);
the untouched test set gives the honest number at the end.

Run: python -m src.tuning      (about 20 minutes)
"""
import json

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from src.config import CONFIG, SEED, path
from src.data import load_split
from src.evaluation import best_threshold, out_of_fold, summarize
from src.models import MODELS, build_pipeline
from src.plots import SERIES, save
from src.tracking import log_candidate, parent_run

TUNING = CONFIG["tuning"]


def sample(trial: optuna.Trial, space: dict) -> dict:
    """Turn the YAML search space into Optuna suggestions."""
    params = {}
    for name, spec in space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def tune(name: str, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
    cfg = TUNING[name]

    def objective(trial: optuna.Trial) -> float:
        params = {**cfg["fixed"], **sample(trial, cfg["space"])}
        _, fold_ap = out_of_fold(build_pipeline(name, **params), X, y)
        log_candidate(f"{name} trial {trial.number}", params,
                      {"AP": fold_ap.mean(), "AP_fold_std": fold_ap.std()}, fold_ap)
        return fold_ap.mean()

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    trials_file = path("tables") / f"optuna_trials_{name}.csv"
    with parent_run(f"optuna_{name}", [trials_file]):
        study.optimize(objective, n_trials=TUNING["n_trials"])
        # Every trial's settings and score, kept as the study's artifact.
        study.trials_dataframe().to_csv(trials_file, index=False)
    return study


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, _, y, _ = load_split()

    rows, best_params, histories = [], {}, {}
    for name in MODELS:
        study = tune(name, X, y)
        best_params[name] = {**TUNING[name]["fixed"], **study.best_params}
        histories[name] = study.trials_dataframe()["value"].cummax()

        # Re-score default and tuned settings the same way, so the table compares like for like.
        for setting, params in [("default", {}), ("tuned", best_params[name])]:
            oof, fold_ap = out_of_fold(build_pipeline(name, **params), X, y)
            rows.append({"model": name, "setting": setting,
                         **summarize(y, oof, best_threshold(y, oof)),
                         "AP": fold_ap.mean(), "AP_fold_std": fold_ap.std()})
        print(f"{name}: best AP {study.best_value:.4f} after {len(study.trials)} trials")

    table = pd.DataFrame(rows)
    table_file = path("tables") / "tuning_results.csv"
    params_file = path("tables") / "best_params.json"
    table.to_csv(table_file, index=False)
    params_file.write_text(json.dumps(best_params, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for color, (name, best_so_far) in zip(SERIES, histories.items()):
        ax.plot(best_so_far.index + 1, best_so_far.values, color=color, linewidth=2, label=name)
    ax.set(xlabel="Optuna trial", ylabel="Best mean CV AP so far")
    ax.set_title("Tuning progress per model", loc="left")
    ax.legend(loc="lower right")
    save(fig, "tuning_history")

    # A separate run that only stores the outcome: comparison table, chosen settings and figure.
    with parent_run("tuning_summary", [table_file, params_file,
                                       path("figures") / "tuning_history.png"]):
        pass
    print()
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
