import numpy as np
from collections import Iterable


def to_list(x):
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, Iterable):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return [x]


def expand_1st(x):
    """
    Adding an extra first dimension

    Args:
        x: (np.array)
    Returns:
         (np.array)
    """
    return np.expand_dims(x, axis=0)
