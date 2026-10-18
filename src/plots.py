"""One plotting style for every figure in the report."""
import matplotlib

matplotlib.use("Agg")  # scripts only save files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import path
from src.evaluation import profit, profit_curve

# Fixed series order: the same model/remedy always gets the same colour across figures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE = SERIES[0]
GREY, INK, MUTED = "#8a8984", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150,
    "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def pr_and_profit(y, scores: dict, name: str) -> None:
    """Left: precision-recall curve per candidate. Right: profit per threshold, dot = best.
    `scores` maps a candidate's label to its out-of-fold probabilities."""
    fig, (ax_pr, ax_profit) = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, (label, oof) in zip(SERIES, scores.items()):
        precision, recall, _ = precision_recall_curve(y, oof)
        ax_pr.plot(recall, precision, color=color, linewidth=2, label=label)
        curve = profit_curve(y, oof)
        ax_profit.plot(curve.index, curve.values, color=color, linewidth=2, label=label)
        ax_profit.plot(curve.idxmax(), curve.max(), "o", color=color, markersize=7)

    ax_pr.axhline(y.mean(), color=GREY, linestyle="--", linewidth=1)
    ax_pr.annotate("random model", (0.02, y.mean() + 0.02), ha="left", color=GREY)
    ax_pr.set(xlabel="Recall (share of subscribers we call)",
              ylabel="Precision (share of calls that subscribe)", ylim=(0, 1))
    ax_pr.set_title("Precision-recall, out-of-fold", loc="left")
    ax_pr.legend(loc="upper right")

    call_all = profit(y, np.ones(len(y), dtype=bool)) * 1000 / len(y)
    ax_profit.axhline(call_all, color=GREY, linestyle="--", linewidth=1)
    ax_profit.annotate("call everyone", (0.98, call_all + 40), ha="right", color=GREY)
    ax_profit.set(xlabel="Threshold: call if score is above", ylabel="EUR per 1,000 clients")
    ax_profit.set_title("Profit by threshold (dot = best)", loc="left")
    ax_profit.legend(loc="upper right")
    save(fig, name)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(path("figures") / f"{name}.png")
    plt.close(fig)
