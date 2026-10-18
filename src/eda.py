"""Step 1 - EDA and split checks.

Everything is computed on the training 80% only: looking at the test set, even just to draw
plots, would let it influence our modelling choices. The only exception is the row-order
(drift) plot, which needs the original row order of the full file and uses no test labels
for any decision we make.

Run: python -m src.eda
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.config import path
from src.data import cv_splitter, load_frame, split_frame
from src.plots import BLUE, GREY, save

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Bins chosen to be readable for a business audience, not optimised for the model.
NUMERIC_BINS = {
    "age": ([17, 24, 29, 39, 49, 59, np.inf], ["18-24", "25-29", "30-39", "40-49", "50-59", "60+"]),
    "balance": ([-np.inf, -1, 500, 2000, np.inf], ["< 0", "0-500", "500-2k", "> 2k"]),
    "campaign": ([0, 1, 2, 3, 5, 10, np.inf], ["1", "2", "3", "4-5", "6-10", "> 10"]),
    "pdays": ([-2, -1, 100, 200, np.inf], ["never", "<= 100 d", "101-200 d", "> 200 d"]),
    "previous": ([-1, 0, 1, 3, np.inf], ["0", "1", "2-3", "> 3"]),
}


def rates(df: pd.DataFrame, by) -> pd.DataFrame:
    return df.groupby(by, observed=True)["y"].agg(n="size", rate="mean")


def rate_bars(ax, table: pd.DataFrame, title: str, base: float) -> None:
    """Horizontal bars of subscription rate, with the overall rate as the reference line."""
    ax.barh(table.index.astype(str), table["rate"], color=BLUE, height=0.6)
    ax.axvline(base, color=GREY, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)


def split_overview(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Rows and subscription rate per part, to show stratification held everywhere."""
    parts = {"full": full, "train": train, "test": test}
    for k, (_, val_idx) in enumerate(cv_splitter().split(train, train["y"]), start=1):
        parts[f"cv fold {k} (validation)"] = train.iloc[val_idx]
    return pd.DataFrame(
        {name: {"rows": len(d), "subscribed": int(d["y"].sum()), "rate": d["y"].mean()}
         for name, d in parts.items()}
    ).T


def column_profile(train: pd.DataFrame) -> pd.DataFrame:
    """No NaNs in this dataset: missing information hides in 'unknown' and pdays = -1."""
    rows = []
    for col in train.columns.drop("y"):
        s = train[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        special = (s == -1).mean() if col == "pdays" else (s == "unknown").mean()
        rows.append({"column": col, "type": "numeric" if numeric else "categorical",
                     "n_unique": s.nunique(), "share_unknown_or_-1": round(special, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    full = load_frame()
    train, test = split_frame(full)
    base = train["y"].mean()

    overview = split_overview(full, train, test)
    profile = column_profile(train)
    categorical = profile.loc[profile["type"] == "categorical", "column"]
    category_rates = pd.concat(
        {col: rates(train, col) for col in categorical}, names=["column", "value"]
    )
    overview.to_csv(path("tables") / "eda_split_overview.csv")
    profile.to_csv(path("tables") / "eda_columns.csv", index=False)
    category_rates.to_csv(path("tables") / "eda_category_rates.csv")

    # Categorical drivers.
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    panels = [("job", "Job", None), ("month", "Month of last contact", MONTHS),
              ("poutcome", "Outcome of previous campaign", None),
              ("contact", "Contact channel", None), ("education", "Education", None),
              ("housing", "Has a housing loan", None)]
    for ax, (col, title, order) in zip(axes.flat, panels):
        t = rates(train, col)
        t = t.reindex(order) if order else t.sort_values("rate", ascending=False)
        rate_bars(ax, t, title, base)
    fig.suptitle(f"Subscription rate by category (training set, dashed line = overall {base:.1%})",
                 x=0.01, ha="left", fontsize=11)
    save(fig, "eda_categorical_rates")

    # Numeric drivers, binned so the shape of the relation is visible.
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    for ax, (col, (bins, labels)) in zip(axes, NUMERIC_BINS.items()):
        rate_bars(ax, rates(train, pd.cut(train[col], bins, labels=labels)), col, base)
    fig.suptitle("Subscription rate by numeric feature (training set, binned)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, "eda_numeric_rates")

    # Leakage evidence: the longer the call, the likelier the "yes", but duration only exists
    # once the call is over, which is exactly the moment we no longer need a prediction.
    fig, ax = plt.subplots(figsize=(5.5, 3))
    t = rates(train, pd.qcut(train["duration"], 5, precision=0))
    t.index = [f"{max(int(i.left), 0)}-{int(i.right)} s" for i in t.index]
    rate_bars(ax, t, "Subscription rate by call duration (quintiles) - dropped", base)
    save(fig, "eda_duration_leakage")

    # Temporal drift: the file is ordered by contact date (May 2008 - Nov 2010) but has no year.
    deciles = pd.qcut(np.arange(len(full)), 10, labels=range(1, 11))
    t = rates(full, deciles)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t.index.astype(int), t["rate"], color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Tenth of the file in original row order (earliest -> latest)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("Subscription rate rises over time", loc="left")
    save(fig, "eda_drift_by_row_order")

    print(overview.round(4).to_string())
    print()
    print(profile.to_string(index=False))
    print(f"\npdays == -1 exactly when previous == 0: "
          f"{((train['pdays'] == -1) == (train['previous'] == 0)).all()}")
    print(f"poutcome == 'unknown' when never contacted: "
          f"{(train.loc[train['pdays'] == -1, 'poutcome'] == 'unknown').mean():.1%}")
    print("Rate by row-order decile:", t["rate"].round(3).tolist())
    print(f"\nFigures -> {path('figures')}\nTables  -> {path('tables')}")


if __name__ == "__main__":
    main()
