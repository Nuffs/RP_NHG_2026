"""
Error Distribution & Co-occurrence Visualization

Reads error_classification JSON and generates figures:
  1. Bar chart — error count per error type
  2. Heatmap — co-occurrence matrix
  3. Per-error co-occurrence bar charts (one figure each)
  4. Histogram — number of error types per query

All figures are saved as PDF/PNG in failure_analysis/results/figures/.

Usage:
    python failure_analysis/plot_errors.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
})

_fa_root = Path(__file__).resolve().parent
INPUT_PATH = _fa_root / "results" / "error_classification_factual.json"
FIGURES_DIR = _fa_root / "results" / "figures"

SHORT_LABELS = {
    "overchunking":         "E1: Overchunking",
    "underchunking":        "E2: Underchunking",
    "E3_context_mismatch":  "E3: Context Mismatch",
    "E4_missed_retrieval":  "E4: Missed Retrieval",
    "E5E6_low_precision":   "E6: Low Ranked",
    "abstention_failure":   "E7: Abstention Failure",
    "E9_hallucination":     "E8: Fabricated Content",
    "E10_incomplete_answer": "E9: Incomplete Answer",
    "E11_misinterpretation": "E10: Misinterpretation",
}

SORT_ORDER = [
    "overchunking",
    "underchunking",
    "E3_context_mismatch",
    "E4_missed_retrieval",
    "E5E6_low_precision",
    "abstention_failure",
    "E9_hallucination",
    "E10_incomplete_answer",
    "E11_misinterpretation",
]

BAR_COLOR = "#4472C4"
BAR_EDGE = "#2F5496"
HEATMAP_CMAP = "Blues"


def _save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", format="png")
    print(f"  ✓ Saved {name}.pdf / .png")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    error_types = data["error_types"]
    per_query = data["per_query"]
    total = data["total_queries"]

    error_keys = [k for k in SORT_ORDER if k in error_types]
    error_keys += [k for k in error_types if k not in error_keys]
    labels = [SHORT_LABELS.get(k, k) for k in error_keys]
    counts = [error_types[k]["count"] for k in error_keys]

    # Co-occurrence matrix
    n = len(error_keys)
    cooccurrence = np.zeros((n, n), dtype=int)

    for pq in per_query:
        errs = set(pq.get("error_types", []))
        for i, ki in enumerate(error_keys):
            if ki not in errs:
                continue
            for j, kj in enumerate(error_keys):
                if kj in errs:
                    cooccurrence[i][j] += 1

    # Figure 1: Error distribution bar chart
    fig1, ax1 = plt.subplots(figsize=(7, 3.5))

    bars = ax1.bar(range(n), counts, color=BAR_COLOR, edgecolor=BAR_EDGE,
                   linewidth=0.5, width=0.65, zorder=3)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{count} ({pct:.0f}%)", ha="center", va="bottom",
                 fontsize=7.5, color="#333333")

    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax1.set_ylabel("Number of Queries")
    ax1.grid(axis="y", color="#dddddd", linestyle="-", alpha=0.7, zorder=0)
    ax1.set_ylim(0, max(counts) * 1.22)
    ax1.set_xlim(-0.5, n - 0.5)

    fig1.tight_layout()
    _save_fig(fig1, "error_distribution")

    # Figure 2: Co-occurrence heatmap
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))

    diag = np.diag(cooccurrence).copy()
    diag[diag == 0] = 1
    cooc_pct = (cooccurrence / diag[:, None] * 100).astype(float)

    im = ax2.imshow(cooc_pct, cmap=HEATMAP_CMAP, aspect="auto", vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            val = cooccurrence[i][j]
            pct = cooc_pct[i][j]
            text_color = "white" if pct > 65 else "#333333"
            if i == j:
                ax2.text(j, i, f"{val}", ha="center", va="center",
                         fontsize=8, fontweight="bold", color=text_color)
            else:
                ax2.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                         fontsize=6.5, color=text_color)

    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=7)

    cbar = fig2.colorbar(im, ax=ax2, shrink=0.82, label="Co-occurrence (%)")
    cbar.ax.tick_params(labelsize=8)

    fig2.tight_layout()
    _save_fig(fig2, "error_cooccurrence_heatmap")

    # Figure 3: Per-error co-occurrence (one figure per error type)
    for idx, key in enumerate(error_keys):
        other_keys = [k for k in error_keys if k != key]
        other_labels = [SHORT_LABELS.get(k, k) for k in other_keys]
        other_counts = [cooccurrence[idx][error_keys.index(k)] for k in other_keys]

        fig_co, ax_co = plt.subplots(figsize=(5, 3.5))

        y_pos = np.arange(len(other_keys))
        ax_co.barh(y_pos, other_counts, color=BAR_COLOR, edgecolor=BAR_EDGE,
                   linewidth=0.4, height=0.55, zorder=3)

        for yp, cnt in zip(y_pos, other_counts):
            if cnt > 0:
                ax_co.text(cnt + 0.3, yp, str(cnt), va="center",
                           fontsize=7, color="#333333")

        ax_co.set_yticks(y_pos)
        ax_co.set_yticklabels(other_labels, fontsize=7)
        ax_co.set_xlabel("Count", fontsize=8)
        label_short = SHORT_LABELS.get(key, key)
        # ax_co.set_title(f"{label_short}  (n = {counts[idx]})", fontsize=9)
        ax_co.grid(axis="x", color="#dddddd", linestyle="-", alpha=0.6, zorder=0)
        ax_co.invert_yaxis()

        fig_co.tight_layout()
        _save_fig(fig_co, f"error_cooccurrence_{key}")
        plt.close(fig_co)

    # Figure 4: Error count distribution per query
    fig4, ax4 = plt.subplots(figsize=(5, 3))

    error_counts_per_query = [pq["error_count"] for pq in per_query]
    max_errors = max(error_counts_per_query) if error_counts_per_query else 0
    hist_counts = [error_counts_per_query.count(i) for i in range(max_errors + 1)]

    bars4 = ax4.bar(range(max_errors + 1), hist_counts, color=BAR_COLOR,
                    edgecolor=BAR_EDGE, linewidth=0.5, width=0.65, zorder=3)

    for bar, cnt in zip(bars4, hist_counts):
        if cnt > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     str(cnt), ha="center", va="bottom",
                     fontsize=8, color="#333333")

    ax4.set_xticks(range(max_errors + 1))
    ax4.set_xticklabels([str(i) for i in range(max_errors + 1)], fontsize=9)
    ax4.set_xlabel("Number of Error Types per Query")
    ax4.set_ylabel("Number of Queries")
    ax4.grid(axis="y", color="#dddddd", linestyle="-", alpha=0.7, zorder=0)

    fig4.tight_layout()
    _save_fig(fig4, "error_count_histogram")

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Displaying plots... Close windows to exit.")
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
