from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from dirtyrag.visualization.data import METHOD_DISPLAY, METHOD_ORDER, order_summary
from dirtyrag.visualization.style import (
    ACCENT_METHOD,
    METRIC_COLOR,
    annotate_bars,
    apply_scientific_style,
    method_color,
    save_figure,
)


METRIC_PANELS = [
    ("strict_success", "Strict Success", "higher is better"),
    ("gold_coverage", "Gold Answer Coverage", "higher is better"),
    ("wrong_leakage", "Wrong-Answer Leakage", "lower is better"),
    ("conflict_sensitivity", "Conflict Sensitivity", "higher is better"),
]


def plot_main_metrics(summary: list[dict[str, Any]], out_dir: Path) -> Path:
    apply_scientific_style()
    rows = order_summary(summary)
    methods = [r["method"] for r in rows]
    labels = [METHOD_DISPLAY.get(m, m) for m in methods]
    n = len(rows)

    if n == 0:
        raise ValueError("No methods found in summary")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (key, title, subtitle) in zip(axes, METRIC_PANELS):
        values = [float(r.get(key, 0.0)) for r in rows]
        colors = [
            METRIC_COLOR[key] if m == ACCENT_METHOD else _shaded(METRIC_COLOR[key], m, methods)
            for m in methods
        ]
        bars = ax.bar(range(n), values, color=colors, edgecolor="#1F2937", linewidth=0.6, width=0.66)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, max(1.0, max(values) * 1.18 if values else 1.0))
        ax.set_title(f"{title}   ({subtitle})", loc="left", pad=8)
        ax.set_ylabel("rate")
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.6, color="#E5E7EB")
        ax.grid(axis="x", visible=False)
        annotate_bars(ax, bars)
        _highlight_ours_tick(ax, methods)

    fig.suptitle(
        "DirtyRAG Main Results on RAMDocs Subset",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    n_examples = rows[0].get("num_examples", "?") if rows else "?"
    fig.text(
        0.5,
        -0.02,
        f"n = {n_examples} cases per method  ·  same LLM, same documents, temperature = 0",
        ha="center",
        fontsize=9,
        color="#6B7280",
    )

    out_path = out_dir / "main_metrics.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_cost_tradeoff(summary: list[dict[str, Any]], out_dir: Path) -> Path:
    apply_scientific_style()
    rows = order_summary(summary)
    if not rows:
        raise ValueError("No methods found in summary")

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for r in rows:
        m = r["method"]
        x = float(r.get("avg_total_tokens") or 0)
        y = float(r.get("strict_success") or 0)
        color = method_color(m)
        is_ours = m == ACCENT_METHOD
        ax.scatter(
            x,
            y,
            s=210 if is_ours else 150,
            color=color,
            edgecolor="#1F2937",
            linewidth=1.0 if is_ours else 0.6,
            marker="D" if is_ours else "o",
            zorder=3,
        )
        ax.annotate(
            METHOD_DISPLAY.get(m, m),
            xy=(x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold" if is_ours else "normal",
            color="#111827",
        )

    ax.set_xlabel("Average total tokens per case  (cost ↗)")
    ax.set_ylabel("Strict Success  (effectiveness ↗)")
    ax.set_title("Cost vs. Effectiveness Trade-off", loc="left")
    ax.set_ylim(-0.02, max(0.6, max(float(r.get("strict_success") or 0) for r in rows) * 1.25))
    xs = [float(r.get("avg_total_tokens") or 0) for r in rows]
    if xs:
        xmax = max(xs) * 1.18 if max(xs) > 0 else 1.0
        ax.set_xlim(0, xmax)
    ax.grid(axis="both", linewidth=0.6, color="#E5E7EB")
    ax.text(
        0.99,
        0.02,
        "upper-left = better",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
        ha="right",
    )

    out_path = out_dir / "cost_tradeoff.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def _shaded(base_color: str, method: str, methods: list[str]) -> str:
    if method == ACCENT_METHOD:
        return base_color
    idx = methods.index(method)
    baselines = [m for m in methods if m != ACCENT_METHOD]
    if not baselines:
        return base_color
    # progressive shading: lighter for earlier baselines, darker as we approach Ours
    rank = baselines.index(method) if method in baselines else 0
    palette = ["#B8C5D6", "#94A6BF", "#6E83A2", "#4F6688"]
    return palette[min(rank, len(palette) - 1)]


def _highlight_ours_tick(ax, methods: list[str]) -> None:
    for i, m in enumerate(methods):
        if m == ACCENT_METHOD:
            label = ax.get_xticklabels()[i]
            label.set_fontweight("bold")
            label.set_color("#B83A4A")
