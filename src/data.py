"""Load Bank Marketing, name the columns, build the target and split off the test set.

Every later step calls `load_split()`, so the split is identical everywhere: it depends only on
the seed in config.yaml, never on which script runs first.
"""
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import CONFIG, SEED, path

# OpenML ships the UCI columns as V1..V16; this restores the documented UCI order.
COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
]

# `duration` is the length of the call we are trying to decide whether to make: it is only
# known after the call, and a 0-second call is always a "no". Keeping it would leak the answer.
LEAKY = ["duration"]


def load_frame() -> pd.DataFrame:
    """Full dataset with readable names and y (1 = subscribed). Still contains `duration`."""
    cfg = CONFIG["data"]
    raw = fetch_openml(data_id=cfg["openml_id"], as_frame=True, data_home=path("data_home")).frame
    df = raw.rename(columns=dict(zip([f"V{i}" for i in range(1, 17)], COLUMNS)))
    df["y"] = (df.pop("Class").astype(str) == cfg["positive_class"]).astype(int)
    # fetch_openml returns categories; plain strings keep later encoders and the API simple.
    for col in df.select_dtypes("category"):
        df[col] = df[col].astype(str)
    return df


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split, stratified on y so both parts keep the 11.7% subscription rate."""
    return train_test_split(
        df, test_size=CONFIG["data"]["test_size"], stratify=df["y"], random_state=SEED
    )


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """X_train, X_test, y_train, y_test with the leaky column removed."""
    train, test = split_frame(load_frame().drop(columns=LEAKY))
    return train.drop(columns="y"), test.drop(columns="y"), train["y"], test["y"]


def cv_splitter() -> StratifiedKFold:
    """The validation scheme: stratified K-fold on the training 80%, same folds for every model."""
    return StratifiedKFold(n_splits=CONFIG["data"]["cv_folds"], shuffle=True, random_state=SEED)
