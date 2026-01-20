import numpy as np
import pytest

from scallops.spots import chastity


@pytest.mark.cli_spot
def test_chastity():
    x = np.expand_dims(
        np.array([[1, 4, 3, 2], [1, 2, 10, 10], [0, 0, 0, 0]]), axis=[2, 3]
    )
    p, p_min = chastity(x)
    np.testing.assert_allclose(p.squeeze(), np.array([4 / 7, 0.5, 0]))
    np.testing.assert_array_equal(p_min.squeeze(), np.array([0]))
