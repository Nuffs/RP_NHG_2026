"""
model_benchmarking/generate_plots.py

Aggregates metrics + RAGChecker scores and produces three publication-ready
figures plus a LaTeX summary table.

Run from repo root:
    python model_benchmarking/generate_plots.py
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
FACTUAL_CSV      = Path("model_benchmarking/results/factual/results_full/metrics.csv")
CLINICAL_CSV     = Path("model_benchmarking/results/clinical/results_full_200/metrics.csv")
FACTUAL_RC_DIR   = Path("model_benchmarking/results/factual/results_full/ragchecker")
CLINICAL_RC_DIR  = Path("model_benchmarking/results/clinical/results_full_200/ragchecker")
PLOTS_DIR        = Path("model_benchmarking/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
TU_CYAN      = "#00A6D6"   # Open-Source
TU_DARK_BLUE = "#0C2340"   # Closed-Source

MODEL_TYPE = {
    "gpt-5.5":         "Closed-Source",
    "gpt-5.4":         "Closed-Source",
    "claude-opus-4-7": "Closed-Source",
    "kimi-k2.6":       "Open-Source",
    "deepseek-v4-pro": "Open-Source",
    "glm-5.1":         "Open-Source",
}

# Mapping from model name to safe filename stem used in ragchecker outputs
SAFE_NAME = {
    "gpt-5.5":         "gpt-55",
    "gpt-5.4":         "gpt-54",
    "claude-opus-4-7": "claude-opus-4-7",
    "kimi-k2.6":       "kimi-k26",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "glm-5.1":         "glm-51",
}

MODELS = list(MODEL_TYPE.keys())

# Display labels for plots
DISPLAY = {
    "gpt-5.5":         "GPT-5.5",
    "gpt-5.4":         "GPT-5.4",
    "claude-opus-4-7": "Claude Opus 4.7",
    "kimi-k2.6":       "Kimi K2.6",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "glm-5.1":         "GLM-5.1",
}

sns.set_style("whitegrid")


# ── Task 1: Data Aggregation ───────────────────────────────────────────────────

def load_metrics_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("model_name").agg(
        avg_cost=("cost_dollars", "mean"),
        avg_speed=("inference_speed_tps", "mean"),
        avg_reasoning_tokens=("reasoning_tokens", "mean"),
    ).reset_index()


def load_ragchecker_scores(rc_dir: Path) -> dict[str, dict]:
    """Returns {model_name: {metric: avg_score}} for all models."""
    scores = {}
    for model_name, safe in SAFE_NAME.items():
        path = rc_dir / f"{safe}_ragchecker.json"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = data["results"]
        metric_keys = list(results[0]["metrics"].keys())
        scores[model_name] = {
            k: float(np.mean([r["metrics"][k] for r in results
                               if r["metrics"].get(k) is not None]))
            for k in metric_keys
        }
    return scores


def build_dataframe() -> pd.DataFrame:
    # ── Metrics CSVs ──────────────────────────────────────────────────────────
    factual_df  = load_metrics_csv(FACTUAL_CSV)
    clinical_df = load_metrics_csv(CLINICAL_CSV)

    fact_agg = aggregate_metrics(factual_df).rename(columns={
        "avg_cost": "factual_avg_cost",
        "avg_speed": "factual_avg_speed",
        "avg_reasoning_tokens": "factual_avg_reasoning",
    })
    clin_agg = aggregate_metrics(clinical_df).rename(columns={
        "avg_cost": "clinical_avg_cost",
        "avg_speed": "clinical_avg_speed",
        "avg_reasoning_tokens": "clinical_avg_reasoning",
    })

    merged = fact_agg.merge(clin_agg, on="model_name", how="outer")

    # Combined averages across both datasets
    merged["avg_cost"]             = (merged["factual_avg_cost"] + merged["clinical_avg_cost"]) / 2
    merged["avg_speed"]            = (merged["factual_avg_speed"] + merged["clinical_avg_speed"]) / 2
    merged["avg_reasoning_tokens"] = (merged["factual_avg_reasoning"] + merged["clinical_avg_reasoning"]) / 2

    # ── RAGChecker scores ──────────────────────────────────────────────────────
    fact_scores = load_ragchecker_scores(FACTUAL_RC_DIR)
    clin_scores = load_ragchecker_scores(CLINICAL_RC_DIR)

    for model in MODELS:
        fs = fact_scores.get(model, {})
        cs = clin_scores.get(model, {})

        merged.loc[merged["model_name"] == model, "factual_f1"]              = fs.get("f1", np.nan) * 100
        merged.loc[merged["model_name"] == model, "clinical_f1"]             = cs.get("f1", np.nan) * 100
        merged.loc[merged["model_name"] == model, "factual_faithfulness"]    = fs.get("faithfulness", np.nan) * 100
        merged.loc[merged["model_name"] == model, "clinical_faithfulness"]   = cs.get("faithfulness", np.nan) * 100
        merged.loc[merged["model_name"] == model, "factual_noise_irrel"]     = fs.get("noise_sensitivity_in_irrelevant", np.nan) * 100
        merged.loc[merged["model_name"] == model, "clinical_noise_irrel"]    = cs.get("noise_sensitivity_in_irrelevant", np.nan) * 100

    merged["combined_f1"]          = (merged["factual_f1"] + merged["clinical_f1"]) / 2
    merged["avg_faithfulness"]     = (merged["factual_faithfulness"] + merged["clinical_faithfulness"]) / 2
    merged["avg_noise_irrel"]      = (merged["factual_noise_irrel"] + merged["clinical_noise_irrel"]) / 2

    # ── Model type & display labels ────────────────────────────────────────────
    merged["model_type"]  = merged["model_name"].map(MODEL_TYPE)
    merged["label"]       = merged["model_name"].map(DISPLAY)

    # Sort: Closed-Source first, then Open-Source
    merged["_sort"] = merged["model_type"].map({"Closed-Source": 0, "Open-Source": 1})
    merged = merged.sort_values(["_sort", "model_name"]).drop(columns="_sort").reset_index(drop=True)

    return merged


# ── Task 2: Plot 1 — Efficiency Trade-offs ─────────────────────────────────────

def plot_efficiency(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Computational Efficiency vs. Clinical Accuracy Trade-offs",
                 fontsize=14, fontweight="bold", y=1.02)

    # Dot size scaled by avg reasoning tokens (min size 80 for models with 0)
    max_rt = df["avg_reasoning_tokens"].max()
    sizes  = 80 + (df["avg_reasoning_tokens"] / max(max_rt, 1)) * 600

    color_map = {"Closed-Source": TU_DARK_BLUE, "Open-Source": TU_CYAN}
    colors    = df["model_type"].map(color_map)

    for ax, (xcol, xlabel, title) in zip(axes, [
        ("avg_cost",  "Average Cost per Query ($)", "Cost vs. Accuracy"),
        ("avg_speed", "Average Inference Speed (tokens/sec)", "Speed vs. Accuracy"),
    ]):
        ax.scatter(df[xcol], df["combined_f1"],
                   s=sizes, c=colors, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)

        for _, row in df.iterrows():
            ax.annotate(row["label"],
                        xy=(row[xcol], row["combined_f1"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8.5, color="#333333")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Combined Overall F1 Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TU_DARK_BLUE,
               markersize=10, label="Closed-Source"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TU_CYAN,
               markersize=10, label="Open-Source"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=6,  label="Small dot = low reasoning"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=14, label="Large dot = high reasoning"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=True)

    plt.tight_layout()
    out = PLOTS_DIR / "plot1_efficiency_tradeoffs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Task 3: Plot 2 — Factual vs Clinical Performance ──────────────────────────

def plot_factual_vs_clinical(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    labels   = df["label"].tolist()
    x        = np.arange(len(labels))
    width    = 0.35

    bars_f = ax.bar(x - width/2, df["factual_f1"],  width, label="Factual Score",
                    color=TU_DARK_BLUE, alpha=0.88)
    bars_c = ax.bar(x + width/2, df["clinical_f1"], width, label="Clinical Score",
                    color=TU_CYAN,      alpha=0.88)

    for bar in list(bars_f) + list(bars_c):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Overall F1 Score (0-100)", fontsize=11)
    ax.set_title("Model Performance: Factual Retrieval vs. Clinical Reasoning",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, max(df[["factual_f1", "clinical_f1"]].max()) * 1.15)
    ax.legend(fontsize=10)

    # Shade closed vs open source regions
    n_closed = (df["model_type"] == "Closed-Source").sum()
    ax.axvspan(-0.5, n_closed - 0.5, alpha=0.04, color=TU_DARK_BLUE, zorder=0)
    ax.axvspan(n_closed - 0.5, len(labels) - 0.5, alpha=0.04, color=TU_CYAN, zorder=0)
    ax.text(n_closed / 2 - 0.5, ax.get_ylim()[1] * 0.97,
            "Closed-Source", ha="center", va="top", fontsize=9,
            color=TU_DARK_BLUE, alpha=0.7)
    ax.text(n_closed + (len(labels) - n_closed) / 2 - 0.5, ax.get_ylim()[1] * 0.97,
            "Open-Source", ha="center", va="top", fontsize=9,
            color=TU_CYAN, alpha=0.7)

    plt.tight_layout()
    out = PLOTS_DIR / "plot2_factual_vs_clinical.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Task 4: Plot 3 — RAG Diagnostics / Safety ─────────────────────────────────

def plot_rag_safety(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = df["label"].tolist()
    x      = np.arange(len(labels))
    width  = 0.35

    bars_f = ax.bar(x - width/2, df["avg_faithfulness"],  width,
                    label="Faithfulness", color=TU_DARK_BLUE, alpha=0.88)
    bars_n = ax.bar(x + width/2, df["avg_noise_irrel"],   width,
                    label="Irrelevant Noise Sensitivity", color="#E63946", alpha=0.75)

    for bar in list(bars_f) + list(bars_n):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score (0-100)", fontsize=11)
    ax.set_title("Generator Safety: Faithfulness vs. Noise Sensitivity",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)

    # Annotation: ideal direction
    ax.annotate("Higher is better", xy=(0.01, 0.92), xycoords="axes fraction",
                fontsize=8, color=TU_DARK_BLUE, style="italic")
    ax.annotate("Lower is better", xy=(0.01, 0.85), xycoords="axes fraction",
                fontsize=8, color="#E63946", style="italic")

    plt.tight_layout()
    out = PLOTS_DIR / "plot3_rag_safety.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Task 5: LaTeX Table ────────────────────────────────────────────────────────

def print_latex_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Model Benchmark Summary: Cost, Speed, and RAGChecker Performance}")
    print(r"\label{tab:model_comparison}")
    print(r"\begin{tabular}{l l r r r r r r}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Type} & \textbf{Avg Cost} & \textbf{Avg Speed} & \textbf{Factual F1} & \textbf{Clinical F1} & \textbf{Combined F1} & \textbf{Avg Reason.} \\")
    print(r" & & \textbf{(\$/query)} & \textbf{(tok/s)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(tokens)} \\")
    print(r"\midrule")

    prev_type = None
    for _, row in df.iterrows():
        if prev_type and row["model_type"] != prev_type:
            print(r"\midrule")
        prev_type = row["model_type"]
        print(
            f"\\texttt{{{row['label']}}} & {row['model_type']} & "
            f"\\${row['avg_cost']:.4f} & "
            f"{row['avg_speed']:.1f} & "
            f"{row['factual_f1']:.1f} & "
            f"{row['clinical_f1']:.1f} & "
            f"{row['combined_f1']:.1f} & "
            f"{row['avg_reasoning_tokens']:.0f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("=" * 70 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building aggregated DataFrame...")
    df = build_dataframe()

    print("\nAggregated summary:")
    print(df[["label", "model_type", "avg_cost", "avg_speed",
              "factual_f1", "clinical_f1", "combined_f1",
              "avg_reasoning_tokens"]].to_string(index=False))

    print("\nGenerating plots...")
    plot_efficiency(df)
    plot_factual_vs_clinical(df)
    plot_rag_safety(df)

    print_latex_table(df)
    print(f"All plots saved to {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
