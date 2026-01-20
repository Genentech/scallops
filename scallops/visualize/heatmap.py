"""SCALLOPS module to plot heatmaps.

This module provides functions for plotting various types of heatmaps.

Functions:
    - `plate`: Plot summary statistics for each well and tile in a heatmap.
    - `base_call_mismatches`: Plot base call mismatches in a heatmap.
    - `in_situ_identity_matrix`: Generate an identity matrix depicting cellular read distribution categorized by read
                                identity.

Example:
    For an example on how to use these functions, refer to the docstrings of individual functions.

Note:
    This module requires the following libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, and `natsort`.
"""

import logging
import warnings
from collections.abc import Callable, Sequence
from itertools import permutations, product
from string import ascii_uppercase
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from natsort import natsorted
from tqdm import tqdm

from scallops.visualize.grid_layout import _grid_indices, _well_grid
from scallops.visualize.grid_layout_constants import SNAKE_BY_ROWS_RIGHT_DOWN
from scallops.visualize.utils import _figsize

logger = logging.getLogger("scallops")


def plate_heatmap(
    df: pd.DataFrame,
    column: str,
    tile_shape: tuple[int, int] = None,
    tile_layout: int = SNAKE_BY_ROWS_RIGHT_DOWN,
    cmap: str | Colormap = "viridis",
    figsize: tuple[float, float] = None,
    colorbar_location: Literal["top", "bottom", "right", "left"] | None = "top",
    vmin_quantile: float = 0,
    vmax_quantile: float = 1,
    missing_value: float = np.nan,
    full_well: Sequence[int] | None = None,
    aggregation_func: Callable | str | None = None,
    overall_ann: dict[str, Callable] | None = None,
    column_names: Sequence[str] | None = None,
    row_names: Sequence[str] | None = None,
):
    """Plot summary statistics for each well and tile in a heatmap.

    This function generates a heatmap to visualize summary statistics for each well and tile
    based on a specified column in the provided DataFrame.

    :param df:
        Data frame to plot. The DataFrame must be grouped by well and tile.
    :param column:
        Column in the data frame to plot.
    :param tile_shape:
        Tile layout rows and columns. If `None`, it is inferred assuming a square layout.
    :param tile_layout:
        Layout constant from `~scallops.visualize.grid_layout`.
    :param cmap:
        The Colormap instance or registered colormap name used to map scalar data to colors.
    :param figsize:
        Figure size. If `None`, it is automatically determined.
    :param colorbar_location:
        Colorbar location. Either 'top' or 'right'.
    :param vmin_quantile:
        Quantile (between 0 and 1) to use for computing the color scale minimum.
    :param vmax_quantile:
        Quantile (between 0 and 1) to use for computing the color scale maximum.
    :param missing_value:
        Value to fill the heatmap if the well and tile combination is not found in the data frame.
    :param full_well:
        Optional sequence of the number of tiles per row in an elliptical well (e.g., Nikon elements tiling).
        If not provided, it is assumed to be squared.
    :param aggregation_func:
        Optional function to aggregate column by if `df` is not aggregated
    :param overall_ann:
        Dictionary of Callables to provide a function for the xlabel per well (e.g., {'Median': np.median}). Must be
        nan resilient.
    :param column_names:
        Optional list of names for columns. Must be ordered by plate layout
    :param row_names:
        Optional list of names for rows. Must be ordered by plate layout

    :return: Returns a Matplotlib Figure object containing the heatmap.

    :example:

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from scallops.visualize import plate_heatmap

        # Create a sample DataFrame
        np.random.seed(42)
        wells = ["A1", "A2", "B1", "B2"]
        tiles = list(range(16))
        w, t = zip(*product(wells, tiles))
        data = pd.DataFrame(
            {
                "well": w,
                "tile": t,
                "value": np.random.normal(size=len(w)),
            }
        ).set_index(["well", "tile"])

        # Generate a heatmap
        plate_heatmap(data, column="value", tile_shape=(4, 4), cmap="viridis")
    """
    if df.index.names[0] != "well" or df.index.names[1] != "tile":
        # If well and tile not in indices, set them
        df = df.reset_index()
        assert "well" in df.columns, "well does not exist in the dataframe"
        assert "tile" in df.columns, "tile does not exist in the dataframe"
        df = df.set_index(["well", "tile"])
    # letters on rows, numbers on columns
    assert df.index.names[0] == "well", "index level 0 must be `well`"
    assert df.index.names[1] == "tile", "index level 1 must be `tile`"
    if aggregation_func is not None:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            df = df.groupby(["well", "tile"])[[column]].agg(aggregation_func)
    else:
        df = df[[column]]
    # Catch the case in which wells are named by numbers with no rows
    title_only = False
    values_0 = df.copy().index.get_level_values(0)
    if (
        pd.api.types.is_numeric_dtype(df.index.dtypes.iloc[0])
        or values_0.str.isdigit().all()
    ):
        title_only = True
        df["well_old"] = values_0.values
        wells = values_0.unique()
        # assuming only 6, 12, 24, 48 and 96 well plates
        all_shapes = {
            3: (1, 3),
            6: (2, 3),
            12: (3, 4),
            24: (4, 6),
            48: (6, 8),
            96: (8, 12),
        }
        nrows, ncols = all_shapes[len(wells)]
        mapp = {
            str(i + 1): f"{x}{y}"
            for i, (x, y) in enumerate(
                product(ascii_uppercase[:nrows], range(1, ncols + 1))
            )
        }
        df["well"] = df.well_old.map(mapp)
        all_wells = natsorted(df.well_old.unique())
        if column_names is not None:
            well_columns = column_names
            if len(column_names) == ncols:
                title_only = False
            elif len(column_names) == ncols * nrows:
                title_only = True
                well_columns_iter = iter(column_names)
            else:
                raise Exception("Column names do not match dimensions")
        else:
            well_columns = all_wells[:ncols]
            well_columns_iter = iter(all_wells)
        if row_names is not None:
            title_only = False
            assert len(row_names) == nrows, "Row names do not match dimensions"
            well_rows = row_names
        else:
            well_rows = [None] * nrows
        if not title_only:
            new_map = dict(
                zip(all_wells, [f"{x}{y}" for x in row_names for y in column_names])
            )
            df["well"] = df.well_old.map(new_map)
            df = (
                df.reset_index(level=0, drop=True)
                .reset_index()
                .set_index(["well", "tile"])
            )
        else:
            well_iter = iter(wells)
    else:
        wells = pd.Series(df.index.get_level_values("well").unique())
        well_columns = natsorted(wells.str[1:].unique())  # e.g. 1
        well_rows = natsorted(wells.str[0].unique())  # e.g. A

    if not pd.api.types.is_numeric_dtype(df.index.dtypes.iloc[1]):
        df.index = df.index.set_levels(df.index.levels[1].astype(int), level="tile")
    ncol = len(well_columns)
    nrow = len(well_rows)
    if tile_shape is None:
        tiles = pd.Series(df.index.get_level_values("tile"))
        tile_dim = int(np.ceil(np.sqrt(max(len(tiles.unique()), tiles.max()))))
        tile_shape = (tile_dim, tile_dim)
    vmin, vmax = df[column].quantile([vmin_quantile, vmax_quantile])
    tile_indices = (
        _well_grid(tile_shape, tuple(full_well), layout=tile_layout)
        if full_well is not None
        else _grid_indices(layout=tile_layout, shape=tile_shape)
    )

    figsize = _figsize(nrow=nrow, ncol=ncol) if figsize is None else figsize
    fig, axes = plt.subplots(
        nrow, ncol, figsize=figsize, squeeze=False, constrained_layout=True
    )
    for well_row_index, well_col_index in product(range(nrow), range(ncol)):
        try:
            well = well_rows[well_row_index] + well_columns[well_col_index]
        except TypeError:
            well = next(well_iter)
        if full_well:
            try:
                single = df.reset_index(level=1).loc[well, ["tile", column]]
            except:
                logger.debug(df.head())
                raise
            data = pd.DataFrame({"tile": tile_indices.astype(int).ravel()})
            data = data.merge(single, how="left", on="tile")[column].values.reshape(
                tile_shape
            )
        else:
            data = np.zeros(tile_shape)
            data[:] = missing_value
            for tile_row_index, tile_col_index in product(
                range(tile_shape[0]), range(tile_shape[1])
            ):
                tile_number = tile_row_index * tile_shape[1] + tile_col_index
                try:
                    color_value = df.loc[(well, tile_number), column]
                    mapped_tile_ij = tile_indices[tile_row_index, tile_col_index]
                    data[mapped_tile_ij[0], mapped_tile_ij[1]] = color_value
                except KeyError:
                    pass
        ax = axes[well_row_index, well_col_index]
        ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

        if well_row_index == 0 and not title_only:
            ax.set_title(well_columns[well_col_index])
        elif title_only:
            title = next(well_columns_iter)
            ax.set_title(title)

        if overall_ann:
            name, func = next(iter(overall_ann.items()))
            ax.set(xlabel=f"{name}: {func(data.ravel())}")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        left_axis = axes[well_row_index, 0]
        left_axis.get_yaxis().set_visible(True)
        left_axis.get_yaxis().set_ticks([])
        if not title_only:
            left_axis.set_ylabel(well_rows[well_row_index], rotation=0, labelpad=10)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    scm = cm.ScalarMappable(norm=norm, cmap=cmap)
    if colorbar_location is not None:
        cbar = fig.colorbar(
            scm,
            ax=axes.ravel().tolist(),
            shrink=0.8,
            location=colorbar_location,
            pad=0.02,
        )
        cbar.set_label(column)
    return fig


def base_call_mismatches_heatmap(
    base_call_mismatches_df: pd.DataFrame,
) -> sns.matrix.ClusterGrid:
    """Plot base call mismatches in a heatmap.

    This function generates a heatmap to visualize base call mismatches in a tabular format. Base call mismatches
    typically occur in sequencing data, where the called bases might differ from the expected or true bases.

    The heatmap is organized to display the counts of base call mismatches across different whitelist bases,
    read positions, and called bases. It provides insights into the patterns of mismatches and their distribution
    within the dataset.

    :param base_call_mismatches_df: Data frame containing base call mismatches. The DataFrame should have columns ['whitelist_base',
        'read_position', 'called_base', 'count'].

    :return: A Seaborn ClusterGrid instance representing the base call mismatches heatmap.

    :example:

    .. code-block:: python

        import pandas as pd
        from scallops.visualize import base_call_mismatches_heatmap

        # Create a sample DataFrame with base call mismatches
        data = {
            "whitelist_base": ["A", "A", "A", "C", "C"],
            "read_position": [1, 2, 1, 3, 2],
            "called_base": ["T", "A", "G", "C", "A"],
            "count": [5, 8, 2, 3, 7],
        }
        base_call_mismatches_df = pd.DataFrame(data)

        # Generate the base call mismatches heatmap
        base_call_mismatches_heatmap(base_call_mismatches_df)
    """
    base_call_mismatches_df = base_call_mismatches_df.pivot(
        index=["whitelist_base", "read_position"], columns="called_base", values="count"
    ).fillna(0)

    base_colors = plt.get_cmap("tab10").colors
    bases = base_call_mismatches_df.index.get_level_values("whitelist_base").unique()
    bases_lut = {bases[i]: base_colors[i % len(base_colors)] for i in range(len(bases))}
    bases_colors = base_call_mismatches_df.index.get_level_values("whitelist_base").map(
        bases_lut
    )
    positions = base_call_mismatches_df.index.get_level_values("read_position").unique()
    position_colors = plt.get_cmap("Greys", len(positions))
    positions_lut = {positions[i]: position_colors(i) for i in range(len(positions))}
    positions_colors = base_call_mismatches_df.index.get_level_values(
        "read_position"
    ).map(positions_lut)

    return sns.clustermap(
        base_call_mismatches_df,
        row_cluster=False,
        col_cluster=False,
        annot=True,
        fmt=",.0f",
        dendrogram_ratio=(0.15, 0),
        row_colors=(positions_colors, bases_colors),
    )


def in_situ_identity_matrix_plot(
    cells_df: pd.DataFrame, xlabel="Total reads", ylabel="Top barcode reads", **kwargs
) -> plt.Axes:
    """Generate an identity matrix depicting cellular read distribution categorized by read
    identity.

    This function creates an identity matrix heatmap to illustrate the distribution of cellular reads categorized by
    read identity. The matrix provides insights into the relationship between different read identity categories,
    particularly focusing on the total reads and top barcode reads.

    :param cells_df: All cell tables from the output cells folder.
    :param xlabel: Set the label for the x-axis of the heatmap.
    :param ylabel: Set the label for the y-axis of the heatmap.
    :param kwargs: Additional keyword arguments are passed through to `matplotlib.pyplot.subplots()`.

    :return: Returns the Axes object with the plot drawn onto it.

    :example:

    .. code-block:: python

        import pandas as pd
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        from scallops.visualize import in_situ_identity_matrix_plot

        np.random.seed(42)
        data = {
            "barcode_match": np.random.choice([True, False], size=50),
            "barcode_0": [f"BC{i}" for i in range(1, 51)],
            "barcode_count_0": np.random.randint(1, 7, size=50),
        }
        data["barcode_count"] = [
            np.random.randint(count, 7) for count in data["barcode_count_0"]
        ]
        cells_df = pd.DataFrame(data)

        # Generate the in-situ identity matrix heatmap
        in_situ_identity_matrix_plot(
            cells_df, xlabel="Total Reads", ylabel="Top Barcode Reads"
        )

        # Show the plot
        plt.show()
    """

    old_output = (
        "barcode" not in cells_df.columns and "cell_barcode_count_0" in cells_df.columns
    )

    idx = cells_df["barcode" if not old_output else "cell_barcode_0"].notna()

    def _obtain_category(
        col: str,
    ) -> pd.Categorical:
        """Represent a column as a categorical variable."""
        values = cells_df.loc[idx, col].values.astype(np.int32)
        values[values >= 5] = 5
        values = values.astype(str)
        cat = pd.Categorical(values=values)
        return cat.rename_categories({"5": "5+"})

    cat1 = _obtain_category(
        "barcode_count" if not old_output else "cell_barcode_count_0"
    )
    cat2 = _obtain_category("barcode_count")
    df = pd.crosstab(cat1, cat2)
    df = df / df.to_numpy().sum()
    df.index.name = ylabel
    df.columns.name = xlabel
    fig, ax = plt.subplots(1, 1, **kwargs)
    sns.heatmap(
        df,
        annot=True,
        fmt=".0%",
        square=True,
        robust=True,
        cmap="Blues",
        cbar=False,
        mask=np.tril(np.ones_like(df, dtype=bool), k=-1),
        ax=ax,
    )
    ax.invert_yaxis()
    ax.tick_params(left=False, bottom=False)
    ax.text(
        0.05,
        0.9,
        f"labels with at least 1 read = {cat1.size:,}",
        transform=ax.transAxes,
    )
    return ax


def plot_well_aggregated_heatmaps(
    df: pd.DataFrame,
    agg_function: callable,
    feature: str,
    grid_size: int = 40,
    plate_shape: tuple[int, int] = (2, 3),
    tile_size_x: float = 2720,
    tile_size_y: float = 2720,
    xlabel: dict = {"Minimum number of spots per cell:": np.nanmin},
    cbar_label: str = "Number of spots",
    x_col_name="x",
    y_col_name="y",
) -> plt.Figure:
    """Generates aggregated heatmap plots for each well in the dataset using a specified aggregation
    function.

    This function creates a figure containing heatmaps for six wells based on the provided DataFrame.
    Each heatmap represents the spatial distribution of a computed metric over a grid, calculated using the supplied aggregation function.

    :param df: Input DataFrame containing data for all wells. Must contain 'well', 'x', and 'y' columns.
    :param agg_function: Aggregation function to apply to each group of data in the grid (e.g., np.mean, np.sum).
                         The function should accept a DataFrame and return a scalar value.
    :param feature: feature to plot. Must exist in dataframe as column
    :param grid_size: Size of the grid (number of bins along x and y axes). Default is 40.
    :param plate_shape: Tuple with row,column shape of the plate
    :param tile_size_x: Size of the tile along the x-axis, used to bin the x coordinates. Default is 2720.
    :param tile_size_y: Size of the tile along the y-axis, used to bin the y coordinates. Default is 2720.
    :param xlabel: Dictionary with label as key and a function to compute a value from the heatmap. Default is {'Minimum number of spots per cell:': np.nanmin}.
    :param cbar_label: Label for the colorbar in the heatmaps. Default is 'Number of spots'.
    :param x_col_name: Column where the x coordinates reside.
    :param y_col_name: Column where the y coordinates reside.
    :return: Matplotlib Figure object containing the heatmaps.

    :example:

        .. code-block:: python

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            # Example DataFrame
            df = pd.DataFrame(
                {
                    "well": np.random.randint(1, 7, size=1000),
                    "x": np.random.rand(1000) * 10000,
                    "y": np.random.rand(1000) * 10000,
                    "value": np.random.rand(1000),
                }
            )


            # Function to aggregate values
            def mean_value(group):
                return group["value"].mean()


            # Generate heatmaps
            fig = plot_well_aggregated_heatmaps(df, mean_value)

            # Display the figure
            plt.show()
    """
    assert feature in df.columns, f"Feature {feature} not in data frame"
    assert "well" in df.columns, "Column well must exist"
    nwells = df.well.nunique()
    assert x_col_name in df.columns and y_col_name in df.columns, (
        "Make sure than both y and x column names are in df"
    )
    nrows, ncols = plate_shape
    assert nwells == nrows * ncols, (
        f"Number of wells in dataframe ({nwells}) does not match the plate shape ({plate_shape})"
    )

    df = df.filter(items=["well", x_col_name, y_col_name, feature])
    df = df.filter(items=["well", x_col_name, y_col_name, feature])

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), layout="constrained"
    )

    # Function to bin cell data into a grid for each well
    def create_heatmap_for_well(df_well, grid_size, tile_size_x, tile_size_y):
        # Initialize the grid with NaNs
        heatmap = np.full((grid_size, grid_size), np.nan)

        # Bin the X and Y coordinates into a grid
        df_well.loc[:, "x_bin"] = (df_well[x_col_name] / tile_size_x).astype(int)
        df_well.loc[:, "y_bin"] = (df_well[y_col_name] / tile_size_y).astype(int)

        # Ensure that the bins are within the grid boundaries
        df_well = df_well[
            (df_well["x_bin"] >= 0)
            & (df_well["x_bin"] < grid_size)
            & (df_well["y_bin"] >= 0)
            & (df_well["y_bin"] < grid_size)
        ]

        # Group by bins and apply the aggregation function
        grouped = df_well.groupby(["x_bin", "y_bin"])
        value = grouped.apply(agg_function, include_groups=False)

        # Fill the heatmap grid with the aggregated values
        for (x_bin, y_bin), val in value.items():
            heatmap[y_bin, x_bin] = val

        return heatmap

    # Create heatmaps for all wells using a list comprehension
    all_heatmaps = [
        create_heatmap_for_well(
            df[df["well"] == i].copy(), grid_size, tile_size_x, tile_size_y
        )
        for i in range(1, 7)
    ]

    # Find the global vmin and vmax for the colorbar, excluding outliers
    vmin = np.nanmin([np.nanquantile(hm, 0.1) for hm in all_heatmaps])
    vmax = np.nanmax([np.nanquantile(hm, 0.9) for hm in all_heatmaps])

    # Loop over the wells and plot each one
    for i, (ax, heatmap) in enumerate(zip(axes.flat, all_heatmaps), start=1):
        # Mask NaNs
        masked_heatmap = np.ma.masked_invalid(heatmap)
        ax.imshow(
            masked_heatmap.T, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(f"Well {i}", fontsize=18)
        label, func = list(xlabel.items())[0]
        val = func(heatmap)
        ax.set_xlabel(f"{label} {val:,.2f}", fontsize=14)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    scm = cm.ScalarMappable(norm=norm, cmap="inferno")
    cbar = fig.colorbar(
        scm, ax=axes.ravel().tolist(), shrink=0.8, location="right", pad=0.02
    )
    cbar.set_label(cbar_label)
    return fig


def plot_barcode_errors(df, title=None, normalize=True, fontsize=12):
    """
    Analyzes barcode errors and plots a heatmap of error rates by position.

    :param df: DataFrame containing barcode data with columns 'barcode', 'closest_match', 'barcode_uncorrected', and
               'mismatches' from the basecall directory `reads` when scallops base calling is run with `--mismatches`
    :param title: Optional title for the plot (default: "Nucleotide Error Rates by Position")
    :param normalize: Whether to normalize error counts to rates (default: True)
    :param fontsize: Base font size for plot labels (default: 12)
    :return: Tuple containing (error_counts_array, matplotlib_figure) or None if validation fails

        :example:

        .. code-block:: python

            import pandas as pd
            import numpy as np

            # Create synthetic data
            np.random.seed(42)
            nucleotides = ["A", "C", "G", "T"]

            # Generate 100 barcodes of length 6
            original = [
                "".join(np.random.choice(nucleotides) for _ in range(6))
                for _ in range(100)
            ]

            # Create uncorrected versions with some errors
            uncorrected = []
            mismatches = []

            for barcode in original:
                # Introduce random errors
                result = list(barcode)
                errors = 0

                for i in range(len(result)):
                    if np.random.random() < 0.1:  # 10% error rate
                        options = [n for n in nucleotides if n != result[i]]
                        result[i] = np.random.choice(options)
                        errors += 1

                uncorrected.append("".join(result))
                mismatches.append(errors)

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "barcode": original,
                    "closest_match": original,
                    "barcode_uncorrected": uncorrected,
                    "mismatches": mismatches,
                }
            )

            # Analyze and plot
            error_counts, fig = analyze_and_plot_barcode_errors(
                df, title="Example Error Analysis"
            )
            plt.show()
    """

    required_columns = ["barcode", "closest_match", "barcode_uncorrected", "mismatches"]
    if df.empty:
        print("Error: Empty DataFrame provided")
        return None

    if not all(col in df.columns for col in required_columns):
        print(
            f"Error: DataFrame is missing required columns. Required: {required_columns}"
        )
        return None

    # Check that all barcodes have the same length
    barcode_lengths = df["barcode"].dropna().str.len().unique()
    if len(barcode_lengths) == 0:
        print("Error: No valid barcodes found")
        return None
    elif len(barcode_lengths) > 1:
        print(f"Error: Barcodes have inconsistent lengths: {barcode_lengths}")
        return None

    nucleotides = ["A", "C", "G", "T"]
    num_positions = barcode_lengths[0]

    # position × original nucleotide × substituted nucleotide
    error_counts = np.zeros((num_positions, len(nucleotides), len(nucleotides)))

    mismatch_df = df[df["mismatches"].notna() & (df["mismatches"] > 0)]

    if len(mismatch_df) == 0:
        print("No barcode errors found in the data")
        return None

    for row in tqdm(
        mismatch_df.itertuples(), total=len(mismatch_df), desc="Processing errors"
    ):
        corrected = getattr(row, "closest_match", None)
        uncorrected = getattr(row, "barcode_uncorrected", None)

        if pd.isna(corrected) or pd.isna(uncorrected) or corrected == uncorrected:
            continue

        if len(corrected) != len(uncorrected):
            continue

        for i, (c1, c2) in enumerate(zip(corrected, uncorrected)):
            if c1 != c2 and c1 in nucleotides and c2 in nucleotides:
                correct_idx = nucleotides.index(c1)
                error_idx = nucleotides.index(c2)
                error_counts[i, correct_idx, error_idx] += 1

    total_errors = error_counts.sum()
    if total_errors == 0:
        print("No valid errors detected in the data")
        return None

    # Calculate error statistics
    error_by_position = error_counts.sum(axis=(1, 2))
    error_types = error_counts.sum(axis=0)

    # Print error statistics
    print(f"Total errors detected: {total_errors:.0f}")
    print("\nErrors by position:")
    for i, count in enumerate(error_by_position):
        print(f"Position {i + 1}: {count:.0f} ({count / total_errors:.1%})")

    print("\nMost common error types:")
    flat_errors = [
        (nucleotides[i], nucleotides[j], error_types[i, j])
        for i, j in permutations(range(len(nucleotides)), 2)
    ]

    # Sort by count and display top errors
    flat_errors.sort(key=lambda x: x[2], reverse=True)
    for orig, err, count in flat_errors[:5]:
        print(f"{orig}→{err}: {count:.0f} ({count / total_errors:.1%})")

    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(
        2,
        num_positions + 1,
        height_ratios=[20, 1],  # Main plots and space for shared label
        width_ratios=[1] * num_positions + [0.1],
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(num_positions)]
    cax = fig.add_subplot(gs[0, -1])

    # Normalize data if requested
    normalized_data = np.zeros_like(error_counts)
    for pos in range(num_positions):
        pos_data = error_counts[pos].copy()
        if normalize and pos_data.sum() > 0:
            normalized_data[pos] = pos_data / pos_data.sum()
        else:
            normalized_data[pos] = pos_data

    vmax = normalized_data.max()

    # Create heatmaps
    for pos in range(num_positions):
        ax = axes[pos]
        pos_data = normalized_data[pos]

        sns.heatmap(
            pos_data,
            annot=False,
            xticklabels=nucleotides,
            yticklabels=nucleotides if pos == 0 else False,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=vmax,
            cbar=False,
        )

        ax.set_title(f"Position {pos + 1}", fontsize=fontsize + 2)
        ax.set_xlabel("")  # Remove individual xlabels since we'll use a shared one
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if pos == 0:
            ax.set_ylabel("Original Nucleotide", fontsize=fontsize + 2)
    cbar = fig.colorbar(ax.collections[0], cax=cax)
    cbar.set_label("Error Rate" if normalize else "Error Count", fontsize=fontsize + 2)
    cbar.ax.tick_params(labelsize=fontsize)
    fig.text(0.5, 0.02, "Substituted Nucleotide", ha="center", fontsize=fontsize + 4)

    if title:
        plt.suptitle(title, y=0.98, fontsize=fontsize + 6)
    else:
        plt.suptitle(
            "Nucleotide Error Rates by Position", y=0.98, fontsize=fontsize + 6
        )

    plt.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )  # Adjust layout to make room for shared xlabel

    return error_counts, fig
