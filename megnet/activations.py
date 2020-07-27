"""
Activation functions used in neural networks
"""
from typing import Callable, Any

import tensorflow.keras.backend as kb
from tensorflow.keras.activations import deserialize, serialize  # noqa
from tensorflow.keras.activations import get as keras_get

from megnet.utils.typing import OptStrOrCallable


def softplus2(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow

    Args:
        x: (Tensor) input tensor

    Returns:
         (Tensor) output tensor
    """
    return kb.relu(x) + kb.log(0.5*kb.exp(-kb.abs(x)) + 0.5)


def get(identifier: OptStrOrCallable = None) -> Callable[..., Any]:
    """
    Get activations by identifier

    Args:
        identifier (str or callable): the identifier of activations

    Returns:
        callable activation

    """
    try:
        return keras_get(identifier)
    except ValueError:
        if isinstance(identifier, str):
            return deserialize(identifier, custom_objects=globals())
    raise ValueError('Could not interpret:', identifier)
