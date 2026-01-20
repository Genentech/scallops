"""Distribution Analysis Submodule.

Provides functions for analyzing and visualizing various distributions in imaging data.



Authors:
    - The SCALLOPS development team
"""

from collections.abc import Sequence
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib import rcParams
from natsort import natsorted
from scipy.interpolate import interp1d
from scipy.stats import t
from sklearn.linear_model import LinearRegression, RANSACRegressor


def ridge_plot(
    data: pd.DataFrame,
    feature: str,
    row: str,
    col: str | None = None,
    scale: str | None = None,
    aspect: int | None = 5,
    height: int | None = 10,
    title: str | None = None,
    palette: str | None = "Spectral",
    **kwargs: dict,
) -> sns.FacetGrid:
    """Create a ridge plot to visualize the distribution of a numeric variable across different
    categorical groups.

    The ridge plot is particularly useful for displaying the distribution of a numeric variable (e.g., gene expression)
    across different categories, such as experimental groups or conditions. It uses kernel density estimation to provide
    a smooth representation of the data.

    :param data:
        A pandas DataFrame containing the feature to plot.
    :param feature:
        The numeric variable in the image to be drawn on separate facets grouped by row and col features.
    :param row:
        The categorical variable in the image to group the feature by and be drawn on separate row facets in the grid.
    :param col:
        The categorical variable in the image to group the feature by and be drawn on separate column facets in the
        grid.
    :param scale:
        Whether to scale `row`. Options are None (no scaling), "standard" (subtract the mean and divide by standard
        deviation), or "log" for log10 scaling.
    :param height:
        The height (in inches) of each facet. See also: aspect.
    :param aspect:
        The aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.
    :param palette:
        The colors to use for the different levels of the hue variable. Should be something that can be interpreted by
        seaborn's color_palette().
    :param title:
        An optional string to set as the title of the figure.
    :param kwargs:
        Optional mapping of arguments to be passed to `FacetGrid`

    :return:
        A seaborn FacetGrid containing the ridge plot.

    :example:

        .. code-block:: python

            import pandas as pd
            import seaborn as sns
            import numpy as np
            import matplotlib.pyplot as plt
            from scallops.visualize import ridge_plot

            # Create a sample DataFrame
            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "category": np.random.choice(["A", "B", "C"], size=300),
                    "group": np.random.choice(["X", "Y"], size=300),
                    "value": np.random.normal(size=300),
                }
            )

            # Generate a ridge plot
            ridge_plot(
                data=data,
                feature="value",
                row="category",
                col="group",
                scale="standard",
                title="Ridge Plot Example",
            )

            # Show the plot
            plt.show()

    .. note::

        - This function uses seaborn's FacetGrid to create a ridge plot, where the density of a numeric variable is
          displayed across different categorical groups.
        - The `scale` parameter allows you to scale the row variable for better visualization.
        - The `palette` parameter determines the color palette to use for different levels of the hue variable.
    """

    old_rcparams = rcParams.copy()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=height)
    pal = sns.color_palette(palette)
    scale_dict = {
        None: lambda x: x,
        "standard": lambda x: (x - np.mean(x)) / np.std(x),
        "log": lambda x: np.log10(x),
    }
    assert scale in scale_dict, f"scale not in {scale_dict.keys()} or None"
    assert row in data.columns, f"{row} not in image columns"
    assert feature in data.columns, f"{feature} not in image columns"
    if col is not None:
        assert col in data.columns, f"{col} was requested but is not in image columns"
    df_ = data.reset_index(drop=True).copy()
    group = [row, col] if col is not None else row
    df_[feature] = df_.groupby(group, group_keys=False)[feature].apply(
        scale_dict[scale]
    )
    g = sns.FacetGrid(
        df_,
        row=row,
        hue=row,
        col=col,
        aspect=aspect,
        height=height,
        palette=pal,
        **kwargs,
    )
    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        feature,
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, feature, clip_on=False, color="w", lw=2, bw_adjust=0.5)
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def _label(x, color, label):
        ax = plt.gca()
        ax.text(
            -0.2,
            0.3,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(_label, row)
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.2)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    if col is not None:
        for ax, comp in zip(g.axes[0], df_[col].unique()):
            ax.set_title(comp, y=0.9)
    lab = f"{feature}" if not scale else f"{scale.capitalize()} {feature}"
    for ax in g.axes[-1]:
        ax.set_xlabel(lab)
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True, right=True, top=True)
    if title is not None:
        g.fig.suptitle(title, y=1.1)
    sns.reset_defaults()
    rcParams.update(old_rcparams)
    return g


def volcano_plot(
    df: pd.DataFrame,
    effect_size_col: str = "∆ AUC",
    ycol: str = "-log10FDR",
    fdr_col: str = "FDR",
    title: str | None = None,
    highlight: dict[str, Sequence[str]]
    | Literal["all", "up", "down"]
    | str
    | None = None,
    highlight_col: str | None = None,
    star: Sequence[str] | None = None,
    top_n: int | None = None,
    bottom_n: int | None = None,
    ax: plt.Axes = None,
    vbar_std: int | float | None = None,
    hbar_value: int | float | None = None,
    magnitude_std: float | None = None,
    legend: tuple[str, str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    logy: bool = False,
    alpha: float = 0.05,
    time_lim: float = 5.0,
    **kwargs: dict,
) -> plt.Axes:
    """Generate a volcano plot to visualize differential expression.

    :param df: The input DataFrame containing the data for the volcano plot.
    :param effect_size_col: The column name representing the effect size (X-axis values).
    :param ycol: The column name representing the y-axis values.
    :param fdr_col: The column name representing the false discovery rate (FDR).
    :param title: The title of the volcano plot.
    :param highlight: A dictionary specifying columns and values to highlight in the plot. Alternatively, 'all' (highlight all up and
        down points), 'up' (only highlight up points) or down (only highlight down points). If a different string it
        assumes dataframe query. Default is None (do not highlight points).
    :param highlight_col: Name of a column in `df` to draw the annotations from. It is required if `highlight` is 'all',
        'up' or 'down'.
    :param star: Star a set of groups (from `highlight_col`), regardless of significance
    :param top_n: The number of top genes to highlight.
    :param bottom_n: The number of bottom genes to highlight.
    :param ax: Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    :param vbar_std: The standard deviation multiplier for vertical bars indicating significance.
        If None, vertical bars will not be plotted.
    :param hbar_value: The horizontal line value for indicating significance.
        If None, no horizontal line will be plotted.
    :param magnitude_std:  Magnitude of the standard deviation for indicating significance.
    :param legend: A tuple specifying legend labels for up-regulated and down-regulated genes.
        If None, no legend will be displayed.
    :param xlim | ylim: Tuples of floats specifying the lower and upper limits for x and y axes. Default is None (range of x and y axes)
    :param logy: If True, apply a logarithmic scale to the y-axis.
    :param alpha: Significance level threshold for highlighting points.
    :param time_lim: Time limit for text adjustment (seconds)
    :param kwargs: Additional keyword arguments to pass to Matplotlib subplots().
    :return: Matplotlib Axes containing the volcano plot.
    :raises ValueError:
        If the specified columns (`effect_size_col` or `ycol`) are not present in the DataFrame.

    :example:

        .. code-block:: python

            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from scallops.visualize import volcano_plot

            # Generate sample data
            np.random.seed(42)

            # Creating a DataFrame with 200 genes
            num_genes = 200
            data = {
                "∆ AUC": np.random.uniform(-0.5, 0.5, num_genes),
                "FDR-BH pval": np.random.uniform(0.05, 1, num_genes),
            }

            # Marking two genes as down-regulated and two as up-regulated
            data["∆ AUC"][:2] = np.random.uniform(-2, -1, 2)  # Down-regulated
            data["FDR-BH pval"][:2] = np.random.uniform(0, 0.01, 2)
            data["∆ AUC"][-2:] = np.random.uniform(1, 2, 2)  # Up-regulated
            data["FDR-BH pval"][-2:] = np.random.uniform(0, 0.01, 2)

            df = pd.DataFrame(data)
            df["-log2FDR"] = df["FDR-BH pval"].apply(lambda x: -np.log(x))
            # Generate volcano plot
            volcano_plot(
                df,
                effect_size_col="∆ AUC",
                ycol="-log2FDR",
                fdr_col="FDR-BH pval",
                vbar_std=2,  # Adjust the standard deviation multiplier for vertical bars
                hbar_value=np.log10(
                    -np.log2(0.05)
                ),  # Adjust the horizontal line value for significance
                legend=("Down-regulated", "Up-regulated"),
            )
    """
    # Validate input DataFrame
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Validate columns existence
    if effect_size_col not in df.columns or ycol not in df.columns:
        raise ValueError("Columns specified not present in the DataFrame")

    # Initialize Axes if not provided
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    # Scatter plot for non-significant points
    effect_size = df[effect_size_col]
    std = magnitude_std if magnitude_std else vbar_std
    assert std is not None, "Either magnitude_std or vbar_std must be provided"
    std *= effect_size.std()
    y = np.log(df[ycol]) if logy else df[ycol]
    ax.scatter(
        x=effect_size, y=y, s=3, label="Not significant", color="grey", alpha=0.5
    )

    # Highlight significant points
    if highlight_col is not None or star is not None:
        assert highlight_col in df.columns, (
            "highlight_col needs to be set with highlight or star options"
        )
    down = df[(effect_size <= -std) & (df[fdr_col] <= alpha)]
    up = df[(effect_size >= std) & (df[fdr_col] <= alpha)]
    ax.scatter(
        x=down[effect_size_col],
        y=np.log(down[ycol]) if logy else down[ycol],
        s=3,
        label=legend[0] if legend is not None else None,
        color="blue",
    )
    ax.scatter(
        x=up[effect_size_col],
        y=np.log(up[ycol]) if logy else up[ycol],
        s=3,
        label=legend[1] if legend is not None else None,
        color="red",
    )
    highligh_df = pd.DataFrame()
    if isinstance(highlight, dict):
        highligh_df = pd.concat(
            [
                df[df[highlight_col].isin(highlight[highlight_col])]
                for highlight_col in highlight.keys()
            ]
        )
    elif isinstance(highlight, str):
        assert highlight_col is not None, (
            "If 'all', 'up' or 'down' are selected, you need to provide the highlight_col parameter, got None"
        )
        if highlight == "up":
            highligh_df = up
        elif highlight == "down":
            highligh_df = down
        else:
            highligh_df = pd.concat((up, down))

    # Highlight requested points via star
    if star is not None:
        assert df[highlight_col].isin(star).any(), (
            f"star value ({star}) not in {highlight_col}"
        )
        starup = set(up[highlight_col]).intersection(star)
        stardo = set(down[highlight_col]).intersection(star)
        starne = set(star).difference(starup.union(stardo))
        if starup:
            st = up.query(f"{highlight_col}.isin(@starup)")
            ax.scatter(
                x=st[effect_size_col],
                y=np.log(st[ycol]) if logy else st[ycol],
                s=25,
                marker="*",
                color="orange",
                zorder=3,
            )
        if stardo:
            st = df.query(f"{highlight_col}.isin(@stardo)")
            ax.scatter(
                x=st[effect_size_col],
                y=np.log(st[ycol]) if logy else st[ycol],
                s=25,
                marker="*",
                color="cyan",
                zorder=3,
            )
        if starne:
            st = df.query(f"{highlight_col}.isin(@starne)")
            ax.scatter(
                x=st[effect_size_col],
                y=np.log(st[ycol]) if logy else st[ycol],
                s=25,
                marker="*",
                color="k",
                zorder=3,
            )
            highligh_df = pd.concat((highligh_df, st))

    # Highlight requested points via top_n/bottom_n
    if top_n is not None:
        missing = set(  # noqa: F841
            df.loc[df[effect_size_col].nlargest(top_n).index, :][highlight_col]
        ).difference(highligh_df[highlight_col])
        st = df.query(f"{highlight_col}.isin(@missing)")
        highligh_df = pd.concat((highligh_df, st))
        ax.scatter(
            x=st[effect_size_col],
            y=np.log(st[ycol]) if logy else st[ycol],
            s=20,
            marker="^",
            color="r",
            zorder=3,
        )
    if bottom_n is not None:
        missing = set(  # noqa: F841
            df.loc[df[effect_size_col].nsmallest(bottom_n).index, :][highlight_col]
        ).difference(highligh_df[highlight_col])
        st = df.query(f"{highlight_col}.isin(@missing)")
        highligh_df = pd.concat((highligh_df, st))
        ax.scatter(
            x=st[effect_size_col],
            y=np.log(st[ycol]) if logy else st[ycol],
            s=10,
            marker="v",
            color="b",
            zorder=3,
        )

    # assign texts for all highlighted points
    texts = [
        ax.text(
            x=r[effect_size_col],
            y=np.log(r[ycol]) if logy else r[ycol],
            s=r[highlight_col],
        )
        for i, r in highligh_df.iterrows()
    ]

    # Set axis labels and parameters
    ax.set_xlabel(effect_size_col, fontdict={"size": 16})
    ax.set_ylabel(f"Log({ycol})" if logy else ycol, fontdict={"size": 16})
    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if vbar_std is not None:
        standard = vbar_std * effect_size.std()
        ax.axvline(-standard, color="grey", linestyle="--", alpha=0.5)
        ax.axvline(standard, color="grey", linestyle="--", alpha=0.5)
    if hbar_value is not None:
        ax.axhline(hbar_value, color="grey", linestyle="--", alpha=0.5)
    ax.spines[["right", "top"]].set_visible(False)

    # Adjust texts to avoid overlapping
    if len(texts) > 0:
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            ax=ax,
            time_lim=time_lim,
            avoid_self=False,
            x=effect_size,
            y=y,
            force_text=(0.5, 0.7),
            expand_axes=True,
        )
    return ax


def _ecdf(sample, extended_min, extended_max):
    sample = np.sort(sample)
    events = np.cumsum(np.ones_like(sample))
    n = sample.size
    cdf = events / n
    if sample[0] > extended_min:
        sample = np.concatenate((np.array([extended_min]), sample))
        cdf = np.concatenate((np.array([0]), cdf))
    if sample[-1] < extended_max:
        sample = np.concatenate((sample, np.array([extended_max])))
        cdf = np.concatenate((cdf, np.array([1])))
    return sample, cdf


def cdf_plot(
    df: pd.DataFrame,
    feature: str,
    targets: Sequence[str] | None,
    groupby_column: str = "gene_symbol_0",
    hue: str | None = None,
    reference_group: str | None = "NTC",
    line_width: int | None = None,
    col: str | None = None,
    col_order: Sequence[Any] | None = None,
    reference_color: str = "grey",
    shade: bool | None = None,
    height: int = 8,
    include_n: bool = True,
) -> Sequence[plt.Axes]:
    """A utility function to create Cumulative Distribution Function (CDF) plots
    comparing a set of conditions against a reference condition. The CDF plots are
    shaded to highlight the differences between the reference and target groups.

    :param df: DataFrame containing the experimental data.
    :param feature: Column representing the feature for CDF plotting.
    :param targets: Target values in groupby_column to plot.
    :param groupby_column: Column used for grouping the data.
    :param reference_group: Reference group for comparison.
    :param line_width: Width of the CDF plot lines.
    :param col: Variable that defines subsets to plot on different columns
    :param col_order: Specify the order column order for categorical levels of `col`.
    :param hue: Variable used to color CDF plots (useful for showing individual guides).
    :param reference_color: Color for the reference group.
    :param shade: Shade the area between target and reference.
    :param height: Figure height.
    :param include_n: Include the sample size per target in legend.
    :return: The axes.
    :example:

        .. code-block:: python

            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from scallops.visualize.distribution import cdf_plot

            # 1. Generate synthetic data
            np.random.seed(42)
            data = []
            groups = {
                "NTC": {"loc": 5, "scale": 1.5, "size": 500},
                "GeneA": {"loc": 8, "scale": 1.5, "size": 500},
                "GeneB": {"loc": 5, "scale": 2.5, "size": 500},
                "GeneC": {"loc": 3, "scale": 1.0, "size": 400},
            }
            for group, params in groups.items():
                values = np.random.normal(**params)
                df_group = pd.DataFrame({"gene": group, "intensity": values})
                data.append(df_group)
            synthetic_df = pd.concat(data, ignore_index=True)

            # 2. Call the cdf_plot function
            axes = cdf_plot(
                df=synthetic_df,
                groupby_column="gene",
                reference_group="NTC",
                feature="intensity",
            )
            plt.show()
    """
    if line_width is None:
        line_width = 4 if hue is None else 2
    if shade is None:
        shade = hue is None

    col_names = None
    if col is not None:
        if col_order is not None:
            col_names = col_order
        else:
            val = df[col]
            if isinstance(val.dtype, pd.CategoricalDtype):
                col_names = val.cat.categories.tolist()
            else:
                col_names = natsorted(val.unique())
    if targets is None:
        target_col = df[groupby_column]
        if isinstance(target_col.dtype, pd.CategoricalDtype):
            targets = target_col.cat.categories.tolist()
        else:
            targets = natsorted(target_col.unique())

        if reference_group is not None:
            targets.remove(reference_group)
    if col_names is not None:
        df = df[df[col].isin(col_names)]

    df = df[
        df[groupby_column].isin(
            targets + [reference_group] if reference_group is not None else []
        )
    ]
    feature_values = df[feature].dropna().values
    # match ranges across all plots
    plot_min = feature_values.min()
    plot_max = feature_values.max()

    ncol = len(col_names) if col_names is not None else 1
    nrow = len(targets)
    figsize = (ncol * height, nrow * height)
    fig, axes = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=figsize,
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    def _cdf_plot(ax, values, color, interpolated_ref_cdf, label):
        n = len(values)
        x, y = _ecdf(values, plot_min, plot_max)

        ax.plot(
            x,
            y,
            color=color,
            label=f"{label} (n = {n:,})" if include_n else label,
            lw=line_width,
            drawstyle="steps-post",
        )

        if interpolated_ref_cdf is not None:
            fillx = np.linspace(x.min(), x.max(), num=500)
            interpolated_fill = interp1d(
                x,
                y,
                kind="previous",
                bounds_error=False,
                fill_value=0,
                assume_sorted=True,
            )
            ax.fill_between(
                fillx,
                interpolated_fill(fillx),
                interpolated_ref_cdf(fillx),
                color=color,
                alpha=0.1,
                step="post",
            )

    palette = sns.color_palette()

    def _plot_target(
        ax, df_target, ylabel, ref_cdf, interpolated_ref_cdf, target_index, show_legend
    ):
        if ref_cdf is not None:
            ax.plot(
                ref_cdf[0],
                ref_cdf[1],
                color=reference_color,
                label=ref_label,
                lw=line_width,
                drawstyle="steps-post",
            )

        if hue is not None:
            # plot individual guides
            index = 0
            for guide, guide_df in df_target.groupby(hue):
                _cdf_plot(
                    ax,
                    guide_df[feature].dropna().values,
                    palette[index % len(palette)],
                    interpolated_ref_cdf,
                    guide,
                )
                index += 1

            ax.set_ylabel(ylabel)
        else:
            c = palette[target_index % len(palette)]
            _cdf_plot(
                ax, df_target[feature].dropna().values, c, interpolated_ref_cdf, ylabel
            )
            ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    for j in range(ncol):
        df_col = df[df[col] == col_names[j]] if col_names is not None else df
        ref_cdf = None
        interpolated_ref_cdf = None

        if reference_group is not None:
            ref_values = (
                df_col[df_col[groupby_column] == reference_group][feature]
                .dropna()
                .values
            )
            ref_label = (
                f"{reference_group} (n = {ref_values.shape[0]:,})"
                if include_n
                else reference_group
            )

            ref_cdf = _ecdf(ref_values, plot_min, plot_max)
            if shade:
                interpolated_ref_cdf = interp1d(
                    ref_cdf[0],
                    ref_cdf[1],
                    kind="previous",
                    bounds_error=False,
                    fill_value=0,
                    assume_sorted=True,
                )

        for i in range(len(targets)):
            df_target = df_col[df_col[groupby_column] == targets[i]]
            show_legend = hue is not None or col_names is not None
            _plot_target(
                axes[i, j],
                df_target,
                targets[i],
                ref_cdf,
                interpolated_ref_cdf,
                i,
                show_legend,
            )

    if hue is None and col is None:  # one shared legend
        labels = []
        handles = []
        for ax_ in axes.flat:
            for handle, label in zip(*ax_.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

        axes[0, ncol - 1].legend(
            handles, labels, loc="center left", bbox_to_anchor=(1, 0.5)
        )
    if col is not None:
        for j in range(ncol):
            axes[0, j].set_title(col_names[j])

    for i in range(nrow - 1):  # not bottom axes
        for j in range(ncol):
            ax = axes[i, j]
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.label.set_visible(False)

    for i in range(nrow):  # not left axes
        for j in range(1, ncol):
            ax = axes[i, j]
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.label.set_visible(False)

    # bottom axes
    for j in range(ncol):
        axes[nrow - 1, j].set_xlabel(feature)
    axes[0, 0].figure.tight_layout()
    return axes


def comparative_effect_scatter(
    df_x: pd.DataFrame,
    x_label: str,
    df_y: pd.DataFrame,
    y_label: str,
    effect_size_column: str,
    highlight_mode: str = "query",
    highlight_query: str | int | dict = 5,
    regression_params: dict | None = None,
    legend_cols: int = 4,
    match_axes: bool = True,
    transform: Callable | None = None,
    colors: Sequence[str] = ("purple", "orange", "blue"),
    top_right_spines: bool = False,
    xeqy: dict | None = None,
    axes_fontsize: int = 20,
    ax: plt.Axes | None = None,
    grouping: str = "group",
    randomseed: int = 42,
    **kwargs,
) -> tuple[plt.Figure | None, plt.Axes, RANSACRegressor | None]:
    """
    Plots and compares the effect sizes of two treatments with advanced highlighting and regression.

    This function creates a scatter plot of effect sizes from two dataframes (e.g., two different
    treatments). It provides two main modes for highlighting points of interest:

    1.  `query`: Highlights points based on independent criteria applied to each dataframe,
        such as the top/bottom N values or a pandas query string.
    2.  `regression`: Fits a linear regression to the data and highlights points based on their
        relationship to the trend. Outliers are defined as points outside a prediction interval (PI),
        while other notable points (e.g., those at the extremes of the trend) are also highlighted.

    A key feature is the consolidated `regression_params` dictionary, which provides a clean,
    extensible interface for controlling the regression analysis and its visual representation.

    :param df_x: DataFrame for the first treatment (x-axis).
    :param x_label: Label for the x-axis.
    :param df_y: DataFrame for the second treatment (y-axis).
    :param y_label: Label for the y-axis.
    :param effect_size_column: The name of the column representing the effect size.
    :param highlight_mode: Method for highlighting. Either 'query' or 'regression'.
    :param highlight_query: Defines what to highlight.
        - If `highlight_mode='query'`: An `int` for top/bottom N points or a `str` query.
        - If `highlight_mode='regression'`: A `dict`, e.g., `{'pi': 0.95, 'n_closest': 10}`. 'pi' sets the prediction
            interval, and 'n_closest' sets the number of points to highlight within the interval.
    :param regression_params: A dictionary to control regression fitting and plotting. If None,
        no regression is performed. Expected keys:
        - 'method' (str): 'ols' (Ordinary Least Squares, default) or 'ransac' (for robust regression).
        - 'residual_threshold' (float): For RANSAC, the maximum residual for a point to be
            considered an inlier. A larger value makes the fit less strict.
        - 'ci_style' (str): 'fill' or 'line' to draw the prediction interval.
        - 'line_kws' (dict): Keyword arguments for styling the regression line (e.g., {'color': 'red'}).
        - 'ci_kws' (dict): Keyword arguments for styling the PI (e.g., {'alpha': 0.1}).
    :param legend_cols: Number of columns in the legend.
    :param match_axes: If True, sets x and y axis limits to be the same.
    :param transform: An optional function to apply to the effect size column.
    :param colors: A list of three colors for highlighting: [Both, X-only, Y-only].
    :param top_right_spines: If False, hides the top and right plot borders.
    :param xeqy: A dictionary of keywords for plotting a y=x line (e.g., {'color': 'grey'}).
    :param axes_fontsize: Font size for the x and y axis labels.
    :param ax: An existing Matplotlib Axes object to plot on. If None, one is created.
    :param grouping: The column name that contains the labels for the points (e.g., "Gene").
    :param randomseed: The random seed for ransac regression.
    :param kwargs: Additional keyword arguments passed to `adjust_text` for label placement.
    :return: A tuple containing the Matplotlib Figure and Axes objects.

    :example:

        .. code-block:: python

            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt

            # Creating sample data
            df_x = pd.DataFrame(
                {
                    "Gene": [f"Gene{i}" for i in range(1, 11)],
                    "Effect_Size": np.random.randn(10),
                }
            )
            df_y = pd.DataFrame(
                {
                    "Gene": [f"Gene{i}" for i in range(1, 11)],
                    "Effect_Size": df_x["Effect_Size"] + np.random.randn(10) * 0.5,
                }
            )

            # Define regression parameters
            regression_config = {
                "method": "ransac",
                "residual_threshold": df_y.effect_size.std(),
                "ci_style": "line",
                "line_kws": {"color": "black", "linestyle": "--", "linewidth": 1.5},
            }

            # Plotting
            fig, ax = comparative_effect_scatter(
                df_x=df_x,
                x_label="Treatment A",
                df_y=df_y,
                y_label="Treatment B",
                effect_size_column="Effect_Size",
                grouping="Gene",
                highlight_mode="regression",
                highlight_query={"pi": 0.95, "n_closest": 4},
                regression_params=regression_config,
                match_axes=True,
                xeqy={"color": "grey", "linestyle": ":"},
            )
            plt.show()
    """
    ransac = None
    if transform is None:

        def transform(x):
            return x

    if not (grouping in df_x.columns or df_x.index.name == grouping) or not (
        grouping in df_y.columns or df_y.index.name == grouping
    ):
        raise ValueError(f"DataFrames must have '{grouping}' as a column or index.")
    df_x, df_y = df_x.copy(), df_y.copy()
    if grouping in df_x.columns:
        df_x = df_x.set_index(grouping)
    if grouping in df_y.columns:
        df_y = df_y.set_index(grouping)
    df_x[effect_size_column] = transform(df_x[effect_size_column])
    df_y[effect_size_column] = transform(df_y[effect_size_column])
    merged_df = pd.merge(
        df_x,
        df_y,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_x", "_y"),
    )
    x_col, y_col = effect_size_column + "_x", effect_size_column + "_y"
    clean_data = merged_df[[x_col, y_col]].dropna()

    genes_x_only, genes_y_only, boths = [], [], []

    if highlight_mode == "regression":
        if not isinstance(highlight_query, dict):
            raise TypeError("For 'regression' mode, highlight_query must be a dict.")
        pi = highlight_query.get("pi", 0.95)
        n_closest = highlight_query.get("n_closest", 10)
        n = len(clean_data)

        if n > 2 and regression_params is not None:
            method = regression_params.get("method", "ols")

            if method == "ransac":
                if RANSACRegressor is None:
                    raise ImportError(
                        "Robust regression requires scikit-learn. Please install it using 'pip install scikit-learn'."
                    )
                residual_threshold = regression_params.get("residual_threshold")
                X = clean_data[x_col].values.reshape(-1, 1)
                y = clean_data[y_col].values
                ransac = RANSACRegressor(
                    LinearRegression(),
                    residual_threshold=residual_threshold,
                    random_state=randomseed,
                ).fit(X, y)
                m, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
                inlier_mask = ransac.inlier_mask_
                pi_base_data = clean_data[inlier_mask]
                residuals_pi = pi_base_data[y_col] - ransac.predict(
                    pi_base_data[[x_col]]
                )
            else:  # OLS
                m, b = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                pi_base_data = clean_data
                residuals_pi = pi_base_data[y_col] - (m * pi_base_data[x_col] + b)

            y_pred = m * clean_data[x_col] + b
            clean_data["residuals"] = clean_data[y_col] - y_pred

            n_pi = len(pi_base_data)
            if n_pi > 2:
                t_val = t.ppf((1 + pi) / 2.0, n_pi - 2)
                s_err = np.sqrt(np.sum(residuals_pi**2) / (n_pi - 2))
                x_mean_pi = np.mean(pi_base_data[x_col])
                ssx_pi = np.sum((pi_base_data[x_col] - x_mean_pi) ** 2)
                pi_half_width = (
                    t_val
                    * s_err
                    * np.sqrt(
                        1 + 1 / n_pi + ((clean_data[x_col] - x_mean_pi) ** 2) / ssx_pi
                    )
                )
                clean_data["pi_lower"], clean_data["pi_upper"] = (
                    y_pred - pi_half_width,
                    y_pred + pi_half_width,
                )

        genes_y_only = clean_data[
            clean_data[y_col] > clean_data.get("pi_upper", np.inf)
        ].index.tolist()
        genes_x_only = clean_data[
            clean_data[y_col] < clean_data.get("pi_lower", -np.inf)
        ].index.tolist()
        within_pi = clean_data.loc[~clean_data.index.isin(genes_x_only + genes_y_only)]
        n_per_side = n_closest // 2
        top_points = within_pi.nlargest(n_per_side, x_col).index
        bottom_points = within_pi.nsmallest(n_per_side, x_col).index
        boths = top_points.union(bottom_points).tolist()

    elif highlight_mode == "query":
        if isinstance(highlight_query, int):
            subset_x = pd.concat(
                [
                    df_x.nlargest(highlight_query, effect_size_column),
                    df_x.nsmallest(highlight_query, effect_size_column),
                ]
            )
            subset_y = pd.concat(
                [
                    df_y.nlargest(highlight_query, effect_size_column),
                    df_y.nsmallest(highlight_query, effect_size_column),
                ]
            )
        else:  # Assumes it's a string query
            subset_x = df_x.query(highlight_query)
            subset_y = df_y.query(highlight_query)

        genes_x_sig = set(subset_x.index)
        genes_y_sig = set(subset_y.index)
        boths = list(genes_x_sig.intersection(genes_y_sig))
        genes_x_only = list(genes_x_sig.difference(boths))
        genes_y_only = list(genes_y_sig.difference(boths))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(
        merged_df[x_col],
        merged_df[y_col],
        s=10,
        marker="o",
        color="grey",
        alpha=0.3,
        zorder=3,
    )

    if len(genes_x_only) > 0:
        ax.scatter(
            merged_df.loc[genes_x_only, x_col],
            merged_df.loc[genes_x_only, y_col],
            color=colors[1],
            label=x_label,
            s=25,
            zorder=4,
            alpha=0.8,
        )
    if len(genes_y_only) > 0:
        ax.scatter(
            merged_df.loc[genes_y_only, x_col],
            merged_df.loc[genes_y_only, y_col],
            color=colors[2],
            label=y_label,
            s=25,
            zorder=4,
            alpha=0.8,
        )
    if len(boths) > 0:
        ax.scatter(
            merged_df.loc[boths, x_col],
            merged_df.loc[boths, y_col],
            color=colors[0],
            label="Both",
            s=25,
            zorder=4,
            alpha=0.8,
        )

    all_annotated = set(boths).union(genes_x_only).union(genes_y_only)
    if all_annotated:
        highlight_df = merged_df.loc[merged_df.index.isin(all_annotated)].reset_index()
        texts = [
            ax.text(r[x_col], r[y_col], r[grouping]) for _, r in highlight_df.iterrows()
        ]
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            **kwargs,
        )

    if (
        highlight_mode == "regression"
        and regression_params is not None
        and "m" in locals()
    ):
        line_kws = regression_params.get("line_kws", {})
        x_vals = np.array(clean_data[x_col].agg(["min", "max"]))
        y_vals = m * x_vals + b
        ax.plot(x_vals, y_vals, **line_kws)

        ci_style = regression_params.get("ci_style")
        if ci_style and "pi_lower" in clean_data.columns:
            sorted_data = clean_data.sort_values(by=x_col)
            ci_kws = regression_params.get("ci_kws", {})
            if ci_style == "fill":
                ci_fill_defaults = {
                    "color": "gray",
                    "alpha": 0.2,
                    "label": f"{int(pi * 100)}% PI",
                }
                ci_fill_defaults.update(ci_kws)
                ax.fill_between(
                    sorted_data[x_col],
                    sorted_data["pi_lower"],
                    sorted_data["pi_upper"],
                    **ci_fill_defaults,
                )
            elif ci_style == "line":
                ci_line_defaults = {
                    "color": "gray",
                    "ls": ":",
                    "lw": 1.5,
                    "alpha": 0.8,
                    "label": f"{int(pi * 100)}% PI",
                }
                ci_line_defaults.update(ci_kws)
                ax.plot(sorted_data[x_col], sorted_data["pi_lower"], **ci_line_defaults)
                ci_line_defaults.pop("label", None)
                ax.plot(sorted_data[x_col], sorted_data["pi_upper"], **ci_line_defaults)
    if match_axes:
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        final_lims = [min(x_lims[0], y_lims[0]), max(x_lims[1], y_lims[1])]
        ax.set_xlim(final_lims)
        ax.set_ylim(final_lims)

    if xeqy is not None:
        lims = ax.get_xlim()
        ax.plot(lims, lims, **xeqy)

    ax.set_xlabel(f"{x_label} {effect_size_column}", fontsize=axes_fontsize)
    ax.set_ylabel(f"{y_label} {effect_size_column}", fontsize=axes_fontsize)
    if legend_cols > 0 and (handles := ax.get_legend_handles_labels())[0]:
        by_label = dict(zip(handles[1], handles[0]))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=legend_cols,
            fancybox=True,
            shadow=True,
        )
    if not top_right_spines:
        ax.spines[["right", "top"]].set_visible(False)
    ax.axvline(0, color="grey", alpha=0.5, ls="--", lw=0.5)
    ax.axhline(0, color="grey", alpha=0.5, ls="--", lw=0.5)

    return fig, ax, ransac


if __name__ == "__main__":
    # For debugging purposes
    np.random.seed(42)
    num_genes = 1000
    # Generate random data
    data = {
        "gene_symbol": [f"gene_{i}" for i in range(num_genes)],
        "∆ AUC": np.random.uniform(-0.5, 0.5, num_genes),
        "FDR-BH pval": np.random.uniform(0.05, 1, num_genes),
    }
    df = pd.DataFrame(data)
    df["-log2FDR"] = -np.log2(df["FDR-BH pval"])
    highlight_genes = ["gene_10", "gene_20", "gene_30"]
    df.loc[df["gene_symbol"].isin(highlight_genes), "FDR-BH pval"] = np.random.uniform(
        0, 0.05, len(highlight_genes)
    )

    # Star specific genes (ensure they exist in the dataset)
    star_genes = ["gene_40", "gene_50", "gene_900", "gene_950"]
    df.loc[df["gene_symbol"].isin(star_genes), "FDR-BH pval"] = np.random.uniform(
        0, 0.05, len(star_genes)
    )

    # Create up-regulated genes
    up_genes = ["gene_900", "gene_901"]
    df.loc[df["gene_symbol"].isin(up_genes), ["∆ AUC", "FDR-BH pval"]] = [
        np.random.uniform(1, 2),
        np.random.uniform(0, 0.01),
    ]
    # Create down-regulated genes
    down_genes = ["gene_950", "gene_951"]
    df.loc[df["gene_symbol"].isin(down_genes), ["∆ AUC", "FDR-BH pval"]] = [
        np.random.uniform(-2, -1),
        np.random.uniform(0, 0.01),
    ]
    # Recalculate -log2FDR for affected genes
    df["-log2FDR"] = -np.log2(df["FDR-BH pval"])

    # Example usage: Pass this DataFrame to your volcano plot function
    volcano_plot(
        df,
        effect_size_col="∆ AUC",
        ycol="-log2FDR",
        fdr_col="FDR-BH pval",
        highlight_col="gene_symbol",
        highlight={"gene_symbol": highlight_genes},
        star=star_genes,
        top_n=2,
        bottom_n=2,
        magnitude_std=2,
    )
