import pandas as pd
from matplotlib import pyplot as plt


def plot_stitch_positions(df: pd.DataFrame, tile_shape: tuple[int, int]) -> plt.Axes:
    """Plot stitch positions.

    :param df: Data frame with stitch positions.
    :param tile_shape: Tile shape
    :return: Axes
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    tiles = df["tile"].values
    y = df["y"].values
    x = df["x"].values

    for i in range(len(x)):
        ax.text(
            x[i] + tile_shape[1] / 2,
            y[i] + tile_shape[0] / 2,
            str(tiles[i]),
            fontsize=6,
            family="monospace",
            va="center",
            ha="center",
        )
    ax.set_xlim(df["x"].min(), (df["x"] + tile_shape[1]).max())
    ax.set_ylim(df["y"].min(), (df["y"] + tile_shape[0]).max())
    ax.invert_yaxis()
    ax.axis("off")
    return ax
