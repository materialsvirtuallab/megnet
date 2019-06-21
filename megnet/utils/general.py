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


def fast_label_binarize(value, labels):
    """Faster version of label binarize

    `label_binarize` from scikit-learn is slow when run 1 label at a time.
    `label_binarize` also is efficient for large numbers of classes, which is not
    common in `megnet`

    Args:
        value: Value to encode
        labels (list): Possible class values
    Returns:
        ([int]): List of integers
    """

    if len(labels) == 2:
        return [int(value == labels[0])]
    else:
        output = [0] * len(labels)
        if value in labels:
            output[labels.index(value)] = 1
        return output
