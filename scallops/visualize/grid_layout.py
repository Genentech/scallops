"""Grid layout constants.

See https://imagej.net/plugins/image-stitching#gridcollection-stitching

Authors:
    - The SCALLOPS development team
"""

from functools import cache
from itertools import islice

import numpy as np

from scallops.visualize.grid_layout_constants import (
    COL_BY_COL_DOWN_LEFT,
    COL_BY_COL_DOWN_RIGHT,
    COL_BY_COL_UP_LEFT,
    COL_BY_COL_UP_RIGHT,
    ROW_BY_ROW_LEFT_DOWN,
    ROW_BY_ROW_LEFT_UP,
    ROW_BY_ROW_RIGHT_DOWN,
    ROW_BY_ROW_RIGHT_UP,
    SNAKE_BY_COLS_DOWN_LEFT,
    SNAKE_BY_COLS_DOWN_RIGHT,
    SNAKE_BY_COLS_UP_LEFT,
    SNAKE_BY_COLS_UP_RIGHT,
    SNAKE_BY_ROWS_LEFT_DOWN,
    SNAKE_BY_ROWS_LEFT_UP,
    SNAKE_BY_ROWS_RIGHT_DOWN,
    SNAKE_BY_ROWS_RIGHT_UP,
)

_col_to_row = dict()
_col_to_row[COL_BY_COL_DOWN_RIGHT] = ROW_BY_ROW_RIGHT_DOWN
_col_to_row[COL_BY_COL_DOWN_LEFT] = ROW_BY_ROW_LEFT_DOWN
_col_to_row[COL_BY_COL_UP_RIGHT] = ROW_BY_ROW_RIGHT_UP
_col_to_row[COL_BY_COL_UP_LEFT] = ROW_BY_ROW_LEFT_UP
_col_to_row[SNAKE_BY_COLS_DOWN_RIGHT] = SNAKE_BY_ROWS_RIGHT_DOWN
_col_to_row[SNAKE_BY_COLS_DOWN_LEFT] = SNAKE_BY_ROWS_LEFT_DOWN
_col_to_row[SNAKE_BY_COLS_UP_RIGHT] = SNAKE_BY_ROWS_RIGHT_UP
_col_to_row[SNAKE_BY_COLS_UP_LEFT] = SNAKE_BY_ROWS_LEFT_UP


def _grid_indices(layout: int, shape: tuple[int, int]) -> np.ndarray:
    """Generates a 2-d array in which the value at i,j contains the index of the row major array
    layout.

    :param layout: Layout constant
    :param shape: Rows and columns of layout
    :return: The grid indices
    """
    column_major = False

    if layout in _col_to_row:
        column_major = True
        layout = _col_to_row[layout]

    nrows, ncols = shape
    indices = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            x = _ij(
                layout,
                i if not column_major else j,
                j if not column_major else i,
                nrows if not column_major else ncols,
                ncols if not column_major else nrows,
            )
            row.append(x)
        indices.append(row)
    return np.array(indices, dtype=[("i", "i4"), ("j", "i4")])


def _ij(layout: int, row: int, col: int, nrows: int, ncols: int) -> tuple[int, int]:
    """Converts a flat index to a 2D grid index (row, column).

    :param layout: int
        The layout type determining how the grid is traversed.
    :param row: int
        The current row in the grid.
    :param col: int
        The current column in the grid.
    :param nrows: int
        The total number of rows in the grid.
    :param ncols: int
        The total number of columns in the grid.
    :return: Tuple[int, int]
        A tuple (i, j) representing the row and column indices.

    :example:
    .. code-block:: python

        # Example usage of _ij
        row, col = _ij(layout, row, col, nrows, ncols)
    """
    if layout == ROW_BY_ROW_RIGHT_DOWN:  # row major order
        pass
    elif layout == ROW_BY_ROW_LEFT_DOWN:
        # column starts at right
        col = ncols - col - 1
    elif layout == ROW_BY_ROW_RIGHT_UP:
        # row starts at bottom
        row = nrows - row - 1
    elif layout == ROW_BY_ROW_LEFT_UP:
        # row starts at bottom
        row = nrows - row - 1
        # column starts at right
        col = ncols - col - 1
    elif layout == SNAKE_BY_ROWS_RIGHT_DOWN:
        # column starts at right if odd row
        if row % 2 == 1:
            col = ncols - col - 1
    elif layout == SNAKE_BY_ROWS_LEFT_DOWN:
        # column starts at right if even row
        if row % 2 == 0:
            col = ncols - col - 1
    elif layout == SNAKE_BY_ROWS_RIGHT_UP:
        # row starts at bottom
        row = nrows - row - 1
        # column starts at right if odd row
        if row % 2 == 1:
            col = ncols - col - 1
    elif layout == SNAKE_BY_ROWS_LEFT_UP:
        # row starts at bottom
        row = nrows - row - 1
        # column starts at right if even row
        if row % 2 == 0:
            col = ncols - col - 1
    else:
        raise ValueError("Invalid layout {}".format(layout))
    return row, col


@cache
def _well_grid(
    shape: tuple, grid_sequence: tuple[int, ...], layout: int = SNAKE_BY_ROWS_RIGHT_DOWN
) -> np.ndarray:
    """Generates a 2D array with the tile number (instead of indices), and -1 in the locations were
    not tiles were captured.

    It takes the `grid_sequence` input, which is a sequence with the number of tiles in that specific row, e.g:
    grid_sequence = [9, 13,17, 19, 19, 21, 21, *[23]*9, 21, 21, 19, 19, 17,13, 9], which means that in the first row,
    only 9 tiles where capture, second 13, etc. This is in line with the Nikon elements structure on a 25x25 grid with
    no overlap

    :param shape: tuple
        The shape of the grid as a tuple (nrows, ncols).
    :param grid_sequence: tuple[int, ...]
        A sequence of integers representing the grid order.
    :param layout: int
        The layout type that determines how the grid is arranged (default is SNAKE_BY_ROWS_RIGHT_DOWN).
    :return: np.ndarray
        A 2D numpy array representing the grid layout, with the tile numbers and -1 for unassigned locations.
    """
    SNAKES = [
        SNAKE_BY_ROWS_RIGHT_DOWN,
        SNAKE_BY_ROWS_LEFT_DOWN,
        SNAKE_BY_ROWS_RIGHT_UP,
        SNAKE_BY_ROWS_LEFT_UP,
        SNAKE_BY_COLS_DOWN_RIGHT,
        SNAKE_BY_COLS_DOWN_LEFT,
        SNAKE_BY_COLS_UP_RIGHT,
        SNAKE_BY_COLS_UP_LEFT,
    ]
    if layout not in range(1, 17):
        raise ValueError("Invalid layout {}".format(layout))
    nrows, ncols = shape
    grid_sequence = np.array(grid_sequence)
    missing_row = False
    if grid_sequence.shape[0] < nrows:
        grid_sequence = np.insert(grid_sequence, [0, grid_sequence.shape[0]], 0)
        missing_row = True
    indices = []
    starts = (ncols - grid_sequence) // 2
    tile_names = iter(range(grid_sequence.sum()))
    for row, col in enumerate(grid_sequence):
        if col == 0:
            cols = [-1] * ncols
        else:
            virtual_row = row - 1 if missing_row else row
            start_idx = starts[row]
            cols = np.ones(ncols) - 2
            cols[start_idx : start_idx + col] = list(islice(tile_names, col))
            if layout in [ROW_BY_ROW_LEFT_DOWN, ROW_BY_ROW_LEFT_UP]:
                cols = cols[::-1]
            elif (layout in SNAKES) and (virtual_row % 2 == 1):
                cols = cols[::-1]
        indices.append(cols)
    indices = np.array(indices)
    if layout in [ROW_BY_ROW_LEFT_UP, ROW_BY_ROW_RIGHT_UP, SNAKE_BY_COLS_DOWN_RIGHT]:
        indices = np.flip(indices, axis=0)
    elif layout in [SNAKE_BY_ROWS_RIGHT_UP]:
        indices = np.flip(indices, axis=1)
    elif layout in [COL_BY_COL_DOWN_RIGHT, SNAKE_BY_ROWS_LEFT_DOWN]:
        indices = indices.T
    elif layout in [COL_BY_COL_DOWN_LEFT, SNAKE_BY_ROWS_LEFT_UP]:
        indices = np.flip(indices.T, axis=1)
    elif layout in [COL_BY_COL_UP_RIGHT, SNAKE_BY_COLS_DOWN_LEFT]:
        indices = np.flip(indices.T, axis=0)
    elif layout in [COL_BY_COL_UP_LEFT, SNAKE_BY_COLS_UP_LEFT]:
        indices = np.flip(np.flip(indices.T, axis=0), axis=1)
    elif layout in [SNAKE_BY_ROWS_LEFT_DOWN]:
        indices = indices.T
    elif layout == SNAKE_BY_COLS_UP_RIGHT:
        indices = np.flip(np.flip(indices, axis=0), axis=1)
    return indices
