"""Feature importance visualizations for backprojection and SHAP-cosine results."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def waterfall_plot(
    df: "pd.DataFrame",
    value_col: str | None = None,
    top_n: int = 20,
    title: str = "Feature importance",
    figsize: tuple[float, float] | None = None,
    positive_color: str = "#2166ac",
    negative_color: str = "#d6604d",
    zero_line_color: str = "0.3",
    feature_col: str = "feature",
    xlabel: str | None = None,
    show_sum: bool = True,
) -> "matplotlib.figure.Figure":
    """Horizontal waterfall / bar chart for ranked feature importance.

    Works with output from both
    :func:`~scallops.features.backprojection.top_features_from_backprojection`
    (column ``score``) and
    :func:`~scallops.features.backprojection.shap_cosine_features`
    (column ``shap`` or ``mean_abs_shap``).

    :param df: DataFrame with at least a feature-name column and a numeric
        value column.  Rows should already be sorted by the caller (they are
        by default from both backprojection functions).
    :param value_col: Name of the numeric column to plot.  Auto-detected from
        ``score``, ``shap``, ``mean_abs_shap`` if *None*.
    :param top_n: Number of features to display (top by ``|value|``).
    :param title: Plot title.
    :param figsize: Figure size ``(width, height)`` in inches.  Defaults to
        ``(8, top_n * 0.35 + 1.5)``.
    :param positive_color: Bar color for positive values.
    :param negative_color: Bar color for negative values.
    :param zero_line_color: Color of the vertical zero line.
    :param feature_col: Column holding feature names.
    :param xlabel: X-axis label.  Auto-set from value_col if *None*.
    :param show_sum: If *True* and the column is ``shap``, annotate the total
        cosine similarity at the top of the plot.
    :return: :class:`matplotlib.figure.Figure`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Auto-detect value column
    if value_col is None:
        for candidate in ("score", "shap", "mean_abs_shap"):
            if candidate in df.columns:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError(
                "Could not auto-detect value column. Pass value_col= explicitly. "
                f"Available columns: {list(df.columns)}"
            )

    if xlabel is None:
        xlabel = {"score": "Backprojection score (z-score units)",
                  "shap": "SHAP value (contribution to cosine similarity)",
                  "mean_abs_shap": "Mean |SHAP| across pairs"}.get(value_col, value_col)

    # Select top_n by |value|
    plot_df = df.copy()
    plot_df["_abs"] = plot_df[value_col].abs()
    plot_df = plot_df.nlargest(top_n, "_abs").reset_index(drop=True)
    # Reverse so largest is at top
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    values  = plot_df[value_col].values.astype(float)
    names   = plot_df[feature_col].values

    colors  = np.where(values >= 0, positive_color, negative_color)

    if figsize is None:
        figsize = (8, max(4, len(values) * 0.38 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(np.arange(len(values)), values, color=colors, edgecolor="none",
                   height=0.7)
    ax.axvline(0, color=zero_line_color, linewidth=0.8, linestyle="--")

    # Feature name labels
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Value annotations on bars
    x_range = np.abs(values).max() if len(values) else 1.0
    for i, (v, bar) in enumerate(zip(values, bars)):
        pad = x_range * 0.02
        ha  = "left" if v >= 0 else "right"
        ax.text(v + (pad if v >= 0 else -pad), i, f"{v:.3f}",
                va="center", ha=ha, fontsize=7, color="0.2")

    # Cosine sum annotation
    if show_sum and value_col == "shap" and "cos_similarity" in df.columns:
        cos_val = df["cos_similarity"].iloc[0]
        ax.text(0.98, 0.01, f"cos(A,B) = {cos_val:.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="0.4",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
