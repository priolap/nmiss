"""
utils/plot.py
-------------
Plotting utilities for visualising ROUGE-NMISS results.
"""

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_average_metrics(
    metrics_df: pd.DataFrame,
    models: List[str],
    metric: str,
    ylabel: str,
    prefix: str,
) -> None:
    """
    Horizontal bar chart comparing baseline vs NMISS scores per difficulty
    level and model.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of ``NMISSScorer.calculate_metrics_for_all_models``.
    models : List[str]
        Model names to include (must match values in the ``MODEL`` column).
    metric : str
        Score component to plot: ``"PREC"``, ``"REC"``, or ``"F1"``.
    ylabel : str
        Label for the x-axis (score axis).
    prefix : str
        ROUGE variant prefix: ``"ROUGE1"``, ``"ROUGE2"``, or ``"ROUGEL"``.
    """
    numeric_cols = metrics_df.select_dtypes(include=np.number).columns
    avg = (
        metrics_df
        .groupby(["MODEL", "LEVEL"])[numeric_cols]
        .mean()
        .reset_index()
    )

    color_palette = plt.get_cmap("tab10")
    width     = 0.8
    n_bars    = len(models) * 2
    bar_width = width / n_bars
    levels    = ["LOW", "MID", "HIGH"]
    y_pos     = np.arange(len(levels))

    plt.figure(figsize=(12, 8))

    for i, model in enumerate(models):
        mdf = avg[avg["MODEL"] == model]
        for j, level in enumerate(levels):
            ldf = mdf[mdf["LEVEL"] == level]
            if ldf.empty:
                continue

            base_val  = float(ldf[f"{prefix}_REF_{metric}"].values[0])
            nmiss_val = float(ldf[f"{prefix}_NMISS_{metric}"].values[0])

            y0 = y_pos[j] - (width / 2) + (2 * i * bar_width)

            plt.barh(
                y0, base_val, bar_width,
                label=f"{model} {prefix} base {metric}",
                color=color_palette(i),
            )
            plt.barh(
                y0 + bar_width, nmiss_val, bar_width,
                label=f"{model} {prefix} NMISS {metric}",
                color=color_palette(i),
                hatch="//",
            )

            plt.text(
                base_val + 0.01, y0 + bar_width / 2,
                f"{base_val:.2f}", va="center", ha="left", fontsize=9,
            )
            plt.text(
                nmiss_val + 0.01, y0 + bar_width + bar_width / 2,
                f"{nmiss_val:.2f}", va="center", ha="left", fontsize=9,
            )

    plt.ylabel("Level")
    plt.xlabel(ylabel)
    plt.title(f"Average {prefix} {metric} — base vs NMISS per level across models")
    plt.yticks(y_pos, ["Low", "Mid", "High"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
