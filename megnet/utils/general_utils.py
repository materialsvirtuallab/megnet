import numpy as np


def to_list(x):
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return x
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