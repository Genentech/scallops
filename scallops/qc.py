"""Quality Control (QC) Module.

This module provides various functionalities for performing quality control on image data.
It includes tools for checking image integrity, validating metadata, and ensuring consistency
across datasets.

Authors:
- The SCALLOPS development team
"""

import seaborn as sns
from matplotlib.pyplot import Axes
from numpy import array, concatenate, repeat
from pandas import DataFrame, crosstab, cut


def plot_mapping_rate(
    allreads_df: DataFrame,
    incell: bool | None = False,
    n_bins: int | None = 50,
    dpi: int | None = 500,
    save_to_file: str | None = None,
) -> Axes | None:
    """Generate plot depicting mapping rate against read quality threshold.

    :param save_to_file: Optional filename to save image to
    :param allreads_df: Dataframe of combined reads
    :param incell: Exclude spots not in cells
    :param n_bins: Number of equal-width bins in the range of n_bins
    :param dpi: Resolution of the figure in dots per inch
    :return: Figure
    """
    df = allreads_df[["Q_min", "match"]].copy()
    if incell:
        df = df[allreads_df["label"] > 0].copy()
    df["bins"] = cut(df["Q_min"], bins=n_bins, right=False)
    cross_bm = crosstab(df["bins"], df["match"])
    cumsum_reverse = cross_bm[::-1].cumsum(axis=0)[::-1]
    sum_per_row = cumsum_reverse.sum(axis=1)
    x = repeat(array([i.left for i in cumsum_reverse.index] + [1.0]), 2)
    series_1 = df.loc[df["Q_min"] >= 1.0, "match"].value_counts()
    y_match = (
        concatenate(
            ((cumsum_reverse[1] / sum_per_row).values, [series_1[1] / series_1.sum()])
        )
        * 100.0
    )
    y_above = (
        concatenate(
            ((sum_per_row / df.shape[0]).values, [series_1.sum() / df.shape[0]])
        )
        * 100.0
    )
    y = concatenate(y_match, y_above)
    labels = ["Exact barcode match"] * y_match.shape[0]
    labels += ["Above quality threshold"] * y_above.shape[0]
    data = DataFrame(
        {
            "Read quality threshold\n(minimum base quality)": x,
            "Reads(%)": y,
            "labels": labels,
        }
    )
    fig = sns.lineplot(
        data=data,
        x="Read quality threshold\n(minimum base quality)",
        y="Reads(%)",
        hue="labels",
    )

    if save_to_file is not None:
        fig.savefig(save_to_file, dpi=dpi)

    return fig
