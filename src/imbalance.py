"""Step 3 - Class imbalance: right metrics, three remedies compared, cost-based threshold.

Only 11.7% of clients subscribe, so a model that never calls anyone is 88% "accurate" and
earns nothing. We therefore judge models by:
- AP (average precision, area under the precision-recall curve): ranking quality on the rare
  class; a random model scores 0.117 here.
- precision and recall at the threshold we would actually use, and the profit that follows.

Remedies compared on LightGBM (the course's working model), all inside 5-fold CV:
1. none          - plain model, 0.5 threshold is just a default
2. class weights - positives weigh n_neg / n_pos (about 7.5x) in the loss
3. SMOTE         - synthetic positives, generated inside each training fold only
Then for each, the threshold is chosen to maximise profit on out-of-fold scores.

Run: python -m src.imbalance
"""
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from src.config import SEED, path
from src.data import load_split
from src.evaluation import best_threshold, out_of_fold, profit_curve, summarize
from src.features import make_preprocessing
from src.plots import GREY, SERIES, save


def remedies(pos_weight: float) -> dict:
    lgbm = dict(random_state=SEED, verbose=-1)
    # Preprocessing steps are unpacked rather than nested: imblearn refuses a Pipeline inside
    # its Pipeline, and keeping all three flat gives them the same step names.
    prep = make_preprocessing().steps
    return {
        "none": Pipeline([*prep, ("model", LGBMClassifier(**lgbm))]),
        "class weights": Pipeline([
            *prep, ("model", LGBMClassifier(scale_pos_weight=pos_weight, **lgbm)),
        ]),
        # imblearn's Pipeline applies SMOTE during fit only, so validation rows are never
        # resampled and synthetic clients are built from training-fold clients only.
        "SMOTE": ImbPipeline([
            *prep, ("smote", SMOTE(random_state=SEED)), ("model", LGBMClassifier(**lgbm)),
        ]),
    }


def main() -> None:
    X, _, y, _ = load_split()
    pos_weight = (y == 0).sum() / (y == 1).sum()  # from training labels only

    rows, scores = [], {}
    for name, pipe in remedies(pos_weight).items():
        oof, fold_ap = out_of_fold(pipe, X, y)
        scores[name] = oof
        for rule, t in [("default 0.5", 0.5), ("profit-optimal", best_threshold(y, oof))]:
            rows.append({"remedy": name, "threshold_rule": rule, "AP_fold_std": fold_ap.std(),
                         **summarize(y, oof, t)})
    table = pd.DataFrame(rows)
    table.to_csv(path("tables") / "imbalance_remedies.csv", index=False)

    # Reference points a manager would ask about.
    call_all = profit_curve(y, pd.Series(1.0, index=y.index).to_numpy()).iloc[0]
    print(f"positive rate {y.mean():.3f} | scale_pos_weight {pos_weight:.2f} | "
          f"profit per 1,000 clients if we call everyone: EUR {call_all:,.0f}\n")
    print(table.round(3).to_string(index=False))

    fig, (ax_pr, ax_profit) = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, (name, oof) in zip(SERIES, scores.items()):
        precision, recall, _ = precision_recall_curve(y, oof)
        ap = table.loc[table["remedy"] == name, "AP"].iloc[0]
        ax_pr.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AP {ap:.3f})")
        curve = profit_curve(y, oof)
        ax_profit.plot(curve.index, curve.values, color=color, linewidth=2, label=name)
        ax_profit.plot(curve.idxmax(), curve.max(), "o", color=color, markersize=7)
    ax_pr.axhline(y.mean(), color=GREY, linestyle="--", linewidth=1)
    ax_pr.annotate("random model", (0.02, y.mean() + 0.02), ha="left", color=GREY)
    ax_pr.set(xlabel="Recall (share of subscribers we call)",
              ylabel="Precision (share of calls that subscribe)", ylim=(0, 1))
    ax_pr.set_title("Precision-recall, out-of-fold", loc="left")
    ax_pr.legend(loc="upper right")
    ax_profit.axhline(call_all, color=GREY, linestyle="--", linewidth=1)
    ax_profit.annotate("call everyone", (0.98, call_all + 40), ha="right", color=GREY)
    ax_profit.set(xlabel="Threshold: call if score is above", ylabel="EUR per 1,000 clients")
    ax_profit.set_title("Profit by threshold (dot = best)", loc="left")
    ax_profit.legend(loc="upper right")
    save(fig, "imbalance_pr_and_profit")


if __name__ == "__main__":
    main()
