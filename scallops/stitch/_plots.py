from collections.abc import Sequence

import fsspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from scallops.cli.util import cli_metadata
from scallops.stitch.shift_utils import _get_overlap_bounding_box


def _max_shift_vs_zncc_plot(
    nccs_before_stitching: np.ndarray,
    shifts_before_stitching: np.ndarray,
    spanning_tree_edges: Sequence[tuple[int, int]],
    pairs: Sequence[tuple[int, int]],
    valid_edges: np.ndarray[bool],
    max_shift: float | None = None,
):
    fig, ax = plt.subplots()

    df = pd.DataFrame(
        data={
            "Valid Edge": valid_edges,
            "ZNCC": nccs_before_stitching,
            "Max shift (µm)": np.max(np.abs(shifts_before_stitching), axis=1),
        }
    )
    in_spanning_tree = None
    if spanning_tree_edges is not None:
        spanning_tree_edges_set = set()
        for edge in spanning_tree_edges:
            spanning_tree_edges_set.add(tuple(sorted(edge)))
        in_spanning_tree = np.zeros(len(nccs_before_stitching), dtype=bool)
        for i in range(len(pairs)):
            in_spanning_tree[i] = tuple(pairs[i]) in spanning_tree_edges_set
        df["In Spanning Tree"] = in_spanning_tree
    sns.scatterplot(
        df.sort_values(
            by=["In Spanning Tree", "ZNCC"] if in_spanning_tree is not None else "ZNCC",
            ascending=True,
        ),
        x="ZNCC",
        y="Max shift (µm)",
        hue="In Spanning Tree" if in_spanning_tree is not None else None,
        style="Valid Edge",
        style_order=[True, False],
        hue_order=[True, False],
        edgecolor="none",
        ax=ax,
    )

    #  ax.set_yscale("log")
    if max_shift is not None:
        ax.axhline(max_shift, c="k", ls=":")
    ax.set_title("ZNCC vs. Max shift (before stitching)")
    return fig


def _shifts_quiver_plot(stage_positions, shifts):
    fig, ax = plt.subplots()
    ax.quiver(
        stage_positions[:, 1],
        stage_positions[:, 0],
        shifts[:, 1],
        shifts[:, 0],
        scale_units="xy",
        angles="xy",
    )
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Tile shifts from stage positions")
    return fig


def _title_plot(
    zncc_val: float,
    radial_correction_k: float | None,
    no_version: bool,
):
    fig = None
    header = []
    if zncc_val is not None:
        header.append(f"ZNCC: {zncc_val:.4f}")
    if radial_correction_k is not None:
        header.append(f"Radial Correction K: {radial_correction_k}")
    if len(header) > 0 or not no_version:
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_ylim((0, 2))
        ax.invert_yaxis()

    if len(header) > 0:
        ax.text(
            0,
            0,
            "\n".join(header),
            va="top",
            ha="left",
        )

    if not no_version:
        md = cli_metadata()

        ax.text(
            0,
            1 if len(header) > 0 else 0,
            f"scallops version: {md['scallops_version']}\n{md['scallops_command']}",
            va="top",
            ha="left",
            wrap=True,
            family="monospace",
        )
    return fig


def _overlapping_regions_plot(
    zncc_values: np.ndarray,
    pairs_after_stitching,
    shifts_after_stitching,
    fractions_after_stitching,
    final_positions,
    tile_shape: tuple[int, int],
    min_overlap_fraction: float = 0,
    vmin: float | None = 0,
    vmax: float | None = 1,
    show_pairs_cutoff: float | None = None,
):
    """Plot overlapping regions.


    :param vmin: Min value to anchor the colormap
    :param vmax: Max value to anchor the colormap
    :return: Matplotlib Axes object
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Reds_r")

    if vmin is None:
        vmin = zncc_values.min()
    if vmax is None:
        vmax = zncc_values.max()

    idx = fractions_after_stitching > min_overlap_fraction
    image_values = (zncc_values - vmin) / (vmax - vmin)
    image_values[image_values > 1] = 1
    image_values[image_values < 0] = 0
    image_values = image_values[idx]
    colors = cmap(image_values)

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    max_width = -np.inf
    max_height = -np.inf
    for i, (pair, shift) in enumerate(
        zip(pairs_after_stitching[idx], shifts_after_stitching[idx])
    ):
        top_left, bottom_right = _get_overlap_bounding_box(
            final_positions[pair[0], 0],
            final_positions[pair[0], 1],
            shift[0],
            shift[1],
            tile_shape[0],
            tile_shape[1],
        )
        x = top_left[1]
        y = top_left[0]
        width = bottom_right[1] - top_left[1]
        height = bottom_right[0] - top_left[0]
        rect = Rectangle(
            (x, y),
            width=width,
            height=height,
            edgecolor=colors[i],
            facecolor=colors[i],
        )

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        ax.add_patch(rect)

    if show_pairs_cutoff is not None:
        for i, (pair, shift) in enumerate(
            zip(pairs_after_stitching[idx], shifts_after_stitching[idx])
        ):
            if zncc_values[i] < show_pairs_cutoff:
                top_left, bottom_right = _get_overlap_bounding_box(
                    final_positions[pair[0], 0],
                    final_positions[pair[0], 1],
                    shift[0],
                    shift[1],
                    tile_shape[0],
                    tile_shape[1],
                )
                x = top_left[1]
                y = top_left[0]
                width = bottom_right[1] - top_left[1]
                height = bottom_right[0] - top_left[0]
                ax.text(
                    x + width / 2,
                    y + height / 2,
                    f"{pair[0]}/{pair[1]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    family="monospace",
                    color="dimgrey",
                )
    scatter = ax.scatter(
        x=[0, 0], y=[0, 0], c=[vmin, vmax], alpha=[0, 0], cmap=cmap
    )  # hidden points for legend
    ax.set_xlim(min_x, max_x + max_width)
    ax.set_ylim(min_y, max_y + max_height)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Overlapping regions ZNCC (after stitching)")
    fig.colorbar(scatter)
    return fig


def _qc_report(
    path: str,
    stage_positions: np.ndarray,
    nccs_before_stitching: np.ndarray | None,
    shifts_before_stitching: np.ndarray | None,
    final_shifts: np.ndarray,
    final_positions: np.ndarray,
    zncc_val: float,
    no_version: bool,
    zncc_values: np.ndarray,
    pairs_after_stitching: np.ndarray | None,
    shifts_after_stitching: np.ndarray | None,
    fractions_after_stitching: np.ndarray | None,
    min_overlap_fraction: float,
    tile_shape: tuple[int, int],
    spanning_tree_edges: list[tuple[int, int]] | None,
    pairs: Sequence[tuple[int, int]],
    valid_edges: np.ndarray[bool] | None,
    max_shift: float | None,
    radial_correction_k: float | None,
):
    fs = fsspec.url_to_fs(path)[0]

    with fs.open(path, "wb") as f:
        with PdfPages(f) as pdf:
            fig = _title_plot(
                zncc_val=zncc_val,
                radial_correction_k=radial_correction_k,
                no_version=no_version,
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            if zncc_values is not None:
                fig = _overlapping_regions_plot(
                    zncc_values=zncc_values,
                    pairs_after_stitching=pairs_after_stitching,
                    shifts_after_stitching=shifts_after_stitching,
                    fractions_after_stitching=fractions_after_stitching,
                    final_positions=final_positions,
                    min_overlap_fraction=min_overlap_fraction,
                    tile_shape=tile_shape,
                )
                pdf.savefig(fig)
                plt.close(fig)
            if final_shifts is not None:
                fig = _shifts_quiver_plot(
                    stage_positions=stage_positions,
                    shifts=final_shifts,
                )
                pdf.savefig(fig)
                plt.close(fig)
            if nccs_before_stitching is not None:
                fig = _max_shift_vs_zncc_plot(
                    nccs_before_stitching=nccs_before_stitching,
                    shifts_before_stitching=shifts_before_stitching,
                    spanning_tree_edges=spanning_tree_edges,
                    pairs=pairs,
                    valid_edges=valid_edges,
                    max_shift=max_shift,
                )
                pdf.savefig(fig)
                plt.close(fig)
