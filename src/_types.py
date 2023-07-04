import numpy as np
from numpy.typing import NDArray


__all__ = ["float_like", "int_like", "bool_like"]

float_like = (
    NDArray[np.float64] |
    NDArray[np.float32] |
    NDArray[np.float16]
)

int_like = (
    NDArray[np.int64] |
    NDArray[np.int32] |
    NDArray[np.int16] |
    NDArray[np.int8]
)

time_like = NDArray[np.datetime64]

bool_like = NDArray[np.bool_]