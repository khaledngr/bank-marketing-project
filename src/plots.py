"""One plotting style for every figure in the report."""
import matplotlib

matplotlib.use("Agg")  # scripts only save files; no window needed
import matplotlib.pyplot as plt

from src.config import path

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


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(path("figures") / f"{name}.png")
    plt.close(fig)
