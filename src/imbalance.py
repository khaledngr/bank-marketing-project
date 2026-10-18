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
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED, path
from src.data import load_split
from src.evaluation import best_threshold, out_of_fold, summarize
from src.features import make_preprocessing
from src.plots import pr_and_profit


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
            rows.append({"remedy": name, "threshold_rule": rule, **summarize(y, oof, t),
                         # CV convention (S5): AP is the mean of the 5 fold scores.
                         "AP": fold_ap.mean(), "AP_fold_std": fold_ap.std()})
    table = pd.DataFrame(rows)
    table.to_csv(path("tables") / "imbalance_remedies.csv", index=False)

    print(f"positive rate {y.mean():.3f} | scale_pos_weight {pos_weight:.2f}\n")
    print(table.round(3).to_string(index=False))

    pr_and_profit(y, scores, "imbalance_pr_and_profit")


if __name__ == "__main__":
    main()
