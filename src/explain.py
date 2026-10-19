"""Step 8 - Explainability: which inputs matter overall, and why one client got their score.

Everything is computed on a held-out CV fold: the tuned LightGBM pipeline is fitted on the other
4 folds and explained on clients it never saw, so the test set stays untouched.

Three global views, because each answers a different question (S8):
- gain importance: how much a feature's splits improved the training objective (training data,
  so it can reward noise);
- permutation importance: how much validation AP drops when the feature is shuffled in the
  frozen model (correlated columns are shuffled together, otherwise they cover for each other);
- mean |SHAP|: how far the feature moves the model's score on average.
Local SHAP waterfalls then explain four clients: a caught subscriber, a missed one, a false
alarm and a clear "no". SHAP values are on the raw score (log-odds), not in probability points.

Run: python -m src.explain
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.config import SEED, path
from src.data import cv_splitter, load_split
from src.models import build_pipeline
from src.plots import SERIES, save
from src.tracking import parent_run

N_REPEATS = 10  # permutation is random: average several shuffles (S8)

# pdays, previous and poutcome all describe the previous campaign (pdays = -1 exactly when
# previous = 0), and our engineered features come from raw columns. Grouping them gives one
# honest number per piece of client information.
SOURCE = {"pdays": "previous contact", "previous": "previous contact",
          "poutcome": "previous contact", "contacted_before": "previous contact",
          "life_stage": "age"}


def source(column: str) -> str:
    return SOURCE.get(column, column)


def model_inputs(columns_step) -> tuple[list[str], list[str]]:
    """For every column the model sees after preprocessing: its readable name and the client
    variable it comes from (one-hot columns map back to their categorical variable)."""
    names, variables = [], []
    for kind, step, cols in columns_step.transformers_:
        if kind == "num":
            names += list(cols)
            variables += list(cols)
        elif kind == "cat":
            for col, categories in zip(cols, step.categories_):
                names += [f"{col}={c}" for c in categories]
                variables += [col] * len(categories)
    return names, variables


def per_variable(shap_values: shap.Explanation, variables: list[str],
                 display: pd.DataFrame) -> shap.Explanation:
    """Sum SHAP values over each variable's model columns (exact, since SHAP is additive) and
    attach the client's real values, so a waterfall reads "unknown = contact", not "1 = contact=unknown"."""
    summed = pd.DataFrame(shap_values.values, columns=variables).T.groupby(level=0, sort=False).sum().T
    return shap.Explanation(values=summed.to_numpy(), base_values=shap_values.base_values,
                            data=display[summed.columns].to_numpy(dtype=object),
                            feature_names=list(summed.columns))


def permutation_importance(pipe, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Drop in validation AP when a group of raw columns is shuffled together."""
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, pipe.predict_proba(X)[:, 1])
    groups: dict[str, list[str]] = {}
    for col in X.columns:
        groups.setdefault(source(col), []).append(col)
    drops = {}
    for group, cols in groups.items():
        losses = []
        for _ in range(N_REPEATS):
            shuffled, order = X.copy(), rng.permutation(len(X))
            for col in cols:  # one shared order keeps grouped columns consistent with each other
                shuffled[col] = X[col].to_numpy()[order]
            losses.append(base - average_precision_score(y, pipe.predict_proba(shuffled)[:, 1]))
        drops[group] = np.mean(losses)
    return pd.Series(drops)


def pick_clients(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Four instructive cases (positions in the validation fold), chosen by a fixed rule."""
    called, yes = scores >= threshold, y.to_numpy() == 1

    def median_of(mask):
        idx = np.flatnonzero(mask)
        return idx[np.argsort(scores[idx])[len(idx) // 2]]

    return {
        "caught subscriber (TP)": median_of(called & yes),
        "missed subscriber (FN)": np.flatnonzero(~called & yes)[np.argmin(scores[~called & yes])],
        "false alarm (FP)": np.flatnonzero(called & ~yes)[np.argmax(scores[called & ~yes])],
        "clear no (TN)": median_of(~called & ~yes),
    }


def main() -> None:
    X, _, y, _ = load_split()
    params = json.loads((path("tables") / "best_params.json").read_text())["lightgbm"]
    tuned = pd.read_csv(path("tables") / "tuning_results.csv")
    threshold = tuned.query("model == 'lightgbm' and setting == 'tuned'")["threshold"].iloc[0]

    train_idx, val_idx = next(cv_splitter().split(X, y))
    pipe = build_pipeline("lightgbm", **params).fit(X.iloc[train_idx], y.iloc[train_idx])
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    scores = pipe.predict_proba(X_val)[:, 1]

    prep, model = Pipeline(pipe.steps[:-1]), pipe.named_steps["model"]
    names, variables = model_inputs(prep.named_steps["columns"])
    sources = [source(v) for v in variables]
    shap_values = shap.TreeExplainer(model)(prep.transform(X_val))
    shap_values.feature_names = names
    # Sanity check: reference + contributions must rebuild the model's raw score (log-odds).
    raw = shap_values.base_values + shap_values.values.sum(axis=1)
    assert np.allclose(1 / (1 + np.exp(-raw)), scores, atol=1e-6)

    # Global importance, all three views aggregated to the same source variables.
    gain = pd.Series(model.booster_.feature_importance("gain"), index=sources).groupby(level=0).sum()
    shap_by_source = pd.DataFrame(shap_values.values, columns=sources).T.groupby(level=0).sum().T
    importance = pd.DataFrame({
        "permutation_AP_drop": permutation_importance(pipe, X_val, y_val),
        "gain_share": gain / gain.sum(),
        "mean_abs_shap": shap_by_source.abs().mean(),
    }).sort_values("permutation_AP_drop", ascending=False)
    importance.to_csv(path("tables") / "importance_global.csv", index_label="feature")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    order = importance.index[::-1]
    titles = ["Permutation: AP lost when shuffled", "Gain: share of training gain",
              "SHAP: mean |contribution| (log-odds)"]
    for ax, column, color, title in zip(axes, importance.columns, SERIES, titles):
        ax.barh(order, importance.loc[order, column], color=color, height=0.6)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    save(fig, "importance_three_views")

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.title("SHAP per client: right = pushes towards 'subscribes'", loc="left")
    save(plt.gcf(), "shap_beeswarm")

    # Local explanations, one contribution per client variable, shown with its real value.
    engineered = prep.named_steps["features"].transform(X_val)
    display = X_val.assign(contacted_before=engineered["contacted_before"],
                           life_stage=engineered["life_stage"])
    local_values = per_variable(shap_values, variables, display)
    rows = []
    for case, i in pick_clients(y_val, scores, threshold).items():
        shap.plots.waterfall(local_values[i], max_display=10, show=False)
        plt.title(f"{case}: P(subscribe) = {scores[i]:.2f}, threshold {threshold:.2f}", loc="left")
        save(plt.gcf(), "shap_local_" + case.split("(")[1].rstrip(")"))
        top = pd.Series(local_values.values[i], index=local_values.feature_names)
        top = top.sort_values(key=abs, ascending=False).head(5)
        rows.append({"case": case, "probability": scores[i], "subscribed": int(y_val.iloc[i]),
                     **X_val.iloc[i].to_dict(),
                     "top_contributions": "; ".join(
                         f"{k}={display.iloc[i][k]} {v:+.2f}" for k, v in top.items())})
    local = pd.DataFrame(rows)
    local.to_csv(path("tables") / "shap_local_examples.csv", index=False)

    figures = ["importance_three_views", "shap_beeswarm"] + [
        "shap_local_" + c for c in ["TP", "FN", "FP", "TN"]]
    with parent_run("explainability", [path("tables") / "importance_global.csv",
                                       path("tables") / "shap_local_examples.csv",
                                       *[path("figures") / f"{f}.png" for f in figures]]):
        pass

    print(f"validation fold: {len(y_val)} clients, AP {average_precision_score(y_val, scores):.3f}\n")
    print(importance.round(4).to_string())
    print()
    print(local[["case", "probability", "subscribed", "top_contributions"]].to_string(index=False))


if __name__ == "__main__":
    main()
