"""
model_benchmarking/plot_cost_vs_speed.py

Generates a poster-quality scatter plot: Average Cost per Query vs.
Average Inference Speed for 6 LLMs, colour-coded by Open/Closed-Source.

Output: model_benchmarking/results/cost_vs_speed_scatter.png  (300 dpi)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH = Path("model_benchmarking/results/metrics.csv")
OUT_PATH = Path("model_benchmarking/results/cost_vs_speed_scatter.png")

# ── Model categorisation ───────────────────────────────────────────────────────
CLOSED_SOURCE = {"gpt-5.5", "gpt-5.4", "claude-opus-4-7"}
OPEN_SOURCE   = {"kimi-k2.6", "deepseek-v4-pro", "glm-5.1"}

# TU Delft palette
COLOR_OPEN   = "#00A6D6"   # TU Delft Cyan
COLOR_CLOSED = "#0C2340"   # TU Delft Dark Blue

# Annotation offsets (x_offset, y_offset) per model — tweak to avoid overlap
OFFSETS: dict[str, tuple[float, float]] = {
    "gpt-5.5":         ( 0.0008,  0.6),
    "gpt-5.4":         ( 0.0008,  0.6),
    "claude-opus-4-7": ( 0.0008, -1.2),
    "kimi-k2.6":       ( 0.0008,  0.6),
    "deepseek-v4-pro": ( 0.0008, -1.2),
    "glm-5.1":         ( 0.0008,  0.6),
}

# ── 1. Load & aggregate ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, dtype={"cost_dollars": float})

agg = (
    df.groupby("model_name")
    .agg(
        avg_cost  =("cost_dollars",       "mean"),
        avg_speed =("inference_speed_tps", "mean"),
    )
    .reset_index()
)

# ── 2. Categorise ──────────────────────────────────────────────────────────────
def categorise(name: str) -> str:
    if name in CLOSED_SOURCE:
        return "Closed-Source"
    if name in OPEN_SOURCE:
        return "Open-Source"
    return "Unknown"

agg["Model Type"] = agg["model_name"].apply(categorise)

palette = {"Closed-Source": COLOR_CLOSED, "Open-Source": COLOR_OPEN}

# ── 3. Plot ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("talk")          # larger fonts for posters

fig, ax = plt.subplots(figsize=(11, 7))

for _, row in agg.iterrows():
    color  = palette[row["Model Type"]]
    marker = "o" if row["Model Type"] == "Closed-Source" else "D"
    ax.scatter(
        row["avg_cost"],
        row["avg_speed"],
        color=color,
        marker=marker,
        s=220,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
    )

# ── 4. Annotations ─────────────────────────────────────────────────────────────
for _, row in agg.iterrows():
    dx, dy = OFFSETS.get(row["model_name"], (0.0008, 0.6))
    color  = palette[row["Model Type"]]
    ax.annotate(
        row["model_name"],
        xy=(row["avg_cost"], row["avg_speed"]),
        xytext=(row["avg_cost"] + dx, row["avg_speed"] + dy),
        fontsize=11,
        color=color,
        fontweight="semibold",
        va="center",
    )

# ── 5. Legend (manual, clean) ──────────────────────────────────────────────────
from matplotlib.lines import Line2D

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CLOSED,
           markersize=12, label="Closed-Source", markeredgecolor="white"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_OPEN,
           markersize=11, label="Open-Source",   markeredgecolor="white"),
]
ax.legend(handles=legend_handles, title="Model Type", title_fontsize=11,
          fontsize=10, framealpha=0.9, loc="upper right")

# ── 6. Axes formatting ─────────────────────────────────────────────────────────
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.3f}"))
ax.set_xlabel("Average Cost per Query (USD)", fontsize=13, labelpad=10)
ax.set_ylabel("Average Inference Speed (Tokens/sec)", fontsize=13, labelpad=10)
ax.set_title(
    "Preliminary Analysis: Cost vs. Speed Trade-off",
    fontsize=15, fontweight="bold", pad=16,
)

# Light grid + spine cleanup for academic look
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

# ── 7. Save ────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PATH}")
plt.show()
