import numpy as np
import pytest

from scallops.visualize.grid_layout import _grid_indices
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

nrows = 3
ncols = 4


@pytest.fixture(params=[False, True])
def by_row(experiment_c, request):
    return request.param


@pytest.mark.utils
def test_right_down(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        ROW_BY_ROW_RIGHT_DOWN if by_row else COL_BY_COL_DOWN_RIGHT, shape
    )
    indices2 = np.array(
        [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (2, 2), (2, 3)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_left_down(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        ROW_BY_ROW_LEFT_DOWN if by_row else COL_BY_COL_DOWN_LEFT, shape
    )
    indices2 = np.array(
        [
            [(0, 3), (0, 2), (0, 1), (0, 0)],
            [(1, 3), (1, 2), (1, 1), (1, 0)],
            [(2, 3), (2, 2), (2, 1), (2, 0)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_right_up(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        ROW_BY_ROW_RIGHT_UP if by_row else COL_BY_COL_UP_RIGHT, shape
    )
    indices2 = np.array(
        [
            [(2, 0), (2, 1), (2, 2), (2, 3)],
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_left_up(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        ROW_BY_ROW_LEFT_UP if by_row else COL_BY_COL_UP_LEFT, shape
    )
    indices2 = np.array(
        [
            [(2, 3), (2, 2), (2, 1), (2, 0)],
            [(1, 3), (1, 2), (1, 1), (1, 0)],
            [(0, 3), (0, 2), (0, 1), (0, 0)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_snake_right_down(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        SNAKE_BY_ROWS_RIGHT_DOWN if by_row else SNAKE_BY_COLS_DOWN_RIGHT, shape
    )
    indices2 = np.array(
        [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(1, 3), (1, 2), (1, 1), (1, 0)],
            [(2, 0), (2, 1), (2, 2), (2, 3)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_snake_left_down(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        SNAKE_BY_ROWS_LEFT_DOWN if by_row else SNAKE_BY_COLS_DOWN_LEFT, shape
    )
    indices2 = np.array(
        [
            [(0, 3), (0, 2), (0, 1), (0, 0)],
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(2, 3), (2, 2), (2, 1), (2, 0)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_snake_right_up(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        SNAKE_BY_ROWS_RIGHT_UP if by_row else SNAKE_BY_COLS_UP_RIGHT, shape
    )
    indices2 = np.array(
        [
            [(2, 0), (2, 1), (2, 2), (2, 3)],
            [(1, 3), (1, 2), (1, 1), (1, 0)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)


@pytest.mark.utils
def test_snake_left_up(by_row):
    shape = (nrows, ncols) if by_row else (ncols, nrows)
    indices1 = _grid_indices(
        SNAKE_BY_ROWS_LEFT_UP if by_row else SNAKE_BY_COLS_UP_LEFT, shape
    )
    indices2 = np.array(
        [
            [(2, 3), (2, 2), (2, 1), (2, 0)],
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(0, 3), (0, 2), (0, 1), (0, 0)],
        ],
        dtype=[("i", "i4"), ("j", "i4")],
    )
    np.testing.assert_equal(indices1, indices2 if by_row else indices2.T)
