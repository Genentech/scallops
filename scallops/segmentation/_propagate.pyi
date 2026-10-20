import numpy as np
from numpy.typing import NDArray

def convert_to_ints(value: float) -> tuple[int, int]: ...
def propagate(
    image: NDArray[np.float64],
    pq: NDArray[np.int32],
    mask: NDArray[np.int8],
    labels: NDArray[np.int32],
    distances: NDArray[np.float64],
    weight: float,
) -> None: ...
