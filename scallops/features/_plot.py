from collections.abc import Sequence

import fsspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def _plot_features(
    df: pd.DataFrame, columns: Sequence[str], output_path: str, centroid_columns
):
    """ " Plot features in df

    :param df: pandas DataFrame
    :param columns: list of column names in df to plot
    :param output_path: PDF output path
    :param centroid_columns: Columns containing coordinates
    """
    fs, output_path = fsspec.url_to_fs(output_path)
    with fs.open(output_path, "wb") as f:
        with PdfPages(f) as pdf:
            for column in columns:
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.ecdfplot(df[column], ax=ax)

                ax.set_xlabel(column)
                ax.set_title(f"{column} CDF")
                pdf.savefig(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 10))
                # plot lower values on top
                df = df.sort_values(column, ascending=False)
                s = 1  # TODO dynamic based on number of points
                s = ax.scatter(
                    df[centroid_columns[0]],
                    df[centroid_columns[1]],
                    c=df[column],
                    s=s,
                    edgecolors="none",
                )
                fig.colorbar(s, ax=ax)
                ax.set_title(column)
                ax.invert_yaxis()
                pdf.savefig(fig)
                plt.close()
