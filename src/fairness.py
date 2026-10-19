"""Step 10 - Fairness audit on age groups and one mitigation (designed on development data).

Sensitive attribute: age, in three life stages that match the U-shape seen in the EDA:
young (< 25, mostly students), working (25-59) and senior (60+, mostly retirees). These are the
same cut-offs as the `life_stage` feature, so the audit looks at the groups the model itself sees.

Fairness definition: equal opportunity. Among clients who WOULD subscribe, every age group should
have the same chance of being called (equal recall / true-positive rate). Demographic parity
(equal call rates) does not fit: subscription rates genuinely differ by age (26% / 10% / 34%),
and with different base rates a calibrated model cannot satisfy every definition at once (S9).
We still report call rate, false-positive rate (unwanted calls) and precision for each group.

Mitigation: post-processing with one threshold per age group, chosen on out-of-fold scores so
each group's recall matches the overall recall. The model itself is unchanged. The thresholds are
saved and frozen; the before/after comparison on the untouched test set happens in the final
evaluation. For contrast we also test "fairness through unawareness" (train without age), which
leaves proxies such as job = retired / student in place.

Run: python -m src.fairness
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.config import path
from src.data import load_split
from src.evaluation import THRESHOLDS, best_threshold, expected_calibration_error, out_of_fold, profit
from src.features import contacted_before, log_skewed, make_preprocessing
from src.models import MODELS, build_pipeline
from src.plots import SERIES, save
from src.tracking import log_candidate, parent_run

GROUPS = ["young <25", "working 25-59", "senior 60+"]
# Same colour for the same setting in every figure (CV and test).
SETTING_COLORS = {"before": SERIES[0], "unawareness": SERIES[1], "after": SERIES[2]}


def age_group(age: pd.Series) -> np.ndarray:
    return pd.cut(age, [0, 24, 59, np.inf], labels=GROUPS).astype(str).to_numpy()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion such as recall (S8): honest error bars for small groups."""
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def group_report(y: pd.Series, scores: np.ndarray, groups: np.ndarray, thresholds: dict) -> pd.DataFrame:
    """Per-group counts and rates at each group's threshold (S8: always show size and positives)."""
    y = np.asarray(y)
    call = scores >= np.array([thresholds[g] for g in groups])
    rows = []
    for g in GROUPS:
        m = groups == g
        yy, cc = y[m], call[m]
        tp = int((cc & (yy == 1)).sum())
        low, high = wilson_interval(tp, int(yy.sum()))
        rows.append({
            "group": g, "threshold": thresholds[g], "clients": int(m.sum()), "subscribers": int(yy.sum()),
            "subscription_rate": yy.mean(), "call_rate": cc.mean(),
            "recall": tp / yy.sum(), "recall_ci_low": low, "recall_ci_high": high,
            "false_positive_rate": (cc & (yy == 0)).sum() / (yy == 0).sum(),
            "precision": tp / max(cc.sum(), 1), "AP": average_precision_score(yy, scores[m]),
            "mean_predicted": scores[m].mean(), "ECE": expected_calibration_error(yy, scores[m]),
        })
    return pd.DataFrame(rows)


def gaps(report: pd.DataFrame) -> dict:
    """Largest minus smallest value across groups: 0 means the groups are treated alike."""
    return {f"{col}_gap": report[col].max() - report[col].min()
            for col in ["recall", "false_positive_rate", "precision", "call_rate"]}


def equal_opportunity_thresholds(y: pd.Series, scores: np.ndarray, groups: np.ndarray,
                                 target_recall: float) -> dict:
    """For each group, the highest threshold that still reaches the target recall."""
    y = np.asarray(y)
    chosen = {}
    for g in GROUPS:
        m = groups == g
        recalls = np.array([((scores[m] >= t) & (y[m] == 1)).sum() / y[m].sum() for t in THRESHOLDS])
        chosen[g] = float(THRESHOLDS[recalls >= target_recall].max())
    return chosen


def plot_groups(table: pd.DataFrame, name: str) -> None:
    """Recall, unwanted calls and call rate per age group, one bar colour per setting."""
    settings = list(table["setting"].unique())
    width = 0.8 / len(settings)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    metrics = [("recall", "Recall: subscribers we call"),
               ("false_positive_rate", "Unwanted calls (FPR)"), ("call_rate", "Share of group called")]
    for ax, (col, title) in zip(axes, metrics):
        for k, setting in enumerate(settings):
            values = table.loc[table["setting"] == setting, col]
            offset = (k - (len(settings) - 1) / 2) * width
            ax.bar(np.arange(len(GROUPS)) + offset, values, width,
                   color=SETTING_COLORS[setting.split(":")[0]], label=setting)
        ax.set_xticks(range(len(GROUPS)), GROUPS)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    # Headroom above 1.0 keeps the legend clear of the bars.
    axes[0].set_ylim(0, 1.35)
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, name)


def main() -> None:
    X, _, y, _ = load_split()
    params = json.loads((path("tables") / "best_params.json").read_text())["lightgbm"]
    groups = age_group(X["age"])

    scores = out_of_fold(build_pipeline("lightgbm", **params), X, y)[0]
    t = best_threshold(y, scores)
    single = {g: t for g in GROUPS}
    target = ((scores >= t) & (y.to_numpy() == 1)).sum() / y.sum()

    # Unawareness: same model without age or the age-derived life_stage feature.
    unaware = Pipeline([*make_preprocessing(steps=(contacted_before, log_skewed)).steps,
                        ("model", MODELS["lightgbm"](**params))])
    unaware_scores = out_of_fold(unaware, X.drop(columns="age"), y)[0]
    t_unaware = best_threshold(y, unaware_scores)

    per_group = equal_opportunity_thresholds(y, scores, groups, target)
    settings = {
        "before: one threshold": (scores, single),
        "unawareness: no age": (unaware_scores, {g: t_unaware for g in GROUPS}),
        "after: age-group thresholds": (scores, per_group),
    }

    reports, summary = [], []
    table_file = path("tables") / "fairness_cv.csv"
    summary_file = path("tables") / "fairness_cv_summary.csv"
    thresholds_file = path("tables") / "fairness_thresholds.json"
    figure = path("figures") / "fairness_cv.png"
    with parent_run("fairness_cv", [table_file, summary_file, thresholds_file, figure]):
        for name, (s, thresholds) in settings.items():
            report = group_report(y, s, groups, thresholds).assign(setting=name)
            call = s >= np.array([thresholds[g] for g in groups])
            overall = {"profit_per_1000": profit(y, call) * 1000 / len(y),
                       "AP": average_precision_score(y, s), **gaps(report)}
            reports.append(report)
            summary.append({"setting": name, **overall})
            # MLflow keys may not contain "<" or "+", so log thresholds as threshold_young etc.
            log_candidate(f"fairness | {name}", {"setting": name, **{
                f"threshold_{g.split()[0]}": v for g, v in thresholds.items()}}, overall)

        table = pd.concat(reports, ignore_index=True)
        table.to_csv(table_file, index=False)
        pd.DataFrame(summary).to_csv(summary_file, index=False)
        thresholds_file.write_text(json.dumps({"target_recall": target, "thresholds": per_group}, indent=2))

        plot_groups(table, "fairness_cv")

    print(f"target recall (overall, one threshold {t}): {target:.3f}  |  age-group thresholds: {per_group}\n")
    cols = ["setting", "group", "threshold", "clients", "subscribers", "subscription_rate", "call_rate",
            "recall", "recall_ci_low", "recall_ci_high", "false_positive_rate", "precision", "ECE"]
    print(table[cols].round(3).to_string(index=False))
    print()
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
