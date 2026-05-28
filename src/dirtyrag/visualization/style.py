from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "direct_llm": "#9CA3AF",
    "vanilla_rag": "#6B7CB8",
    "relevance_filter_rag": "#4F8EC1",
    "crag_style_rag": "#3A6EA5",
    "evidenceboard_rag": "#D1495B",
}

ACCENT_METHOD = "evidenceboard_rag"

METRIC_COLOR = {
    "strict_success": "#2E7D5B",
    "gold_coverage": "#3A6EA5",
    "wrong_leakage": "#C0392B",
    "conflict_sensitivity": "#B8860B",
}


def apply_scientific_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def method_color(method: str) -> str:
    return PALETTE.get(method, "#777777")


def annotate_bars(ax, bars, *, fmt: str = "{:.2f}", offset: float = 0.012) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + span * offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )


def save_figure(fig: plt.Figure, out_path) -> None:
    fig.savefig(out_path, dpi=300)
    pdf_path = out_path.with_suffix(".pdf")
    try:
        fig.savefig(pdf_path, dpi=300)
    except Exception:
        pass
