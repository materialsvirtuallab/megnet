from typing import Union, Callable

import tensorflow.keras.backend as kb
from tensorflow.keras.activations import get as keras_get
from tensorflow.keras.activations import deserialize, serialize  # serialize is needed here


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


def get(identifier: Union[str, Callable]) -> Callable:
    try:
        return keras_get(identifier)
    except ValueError:
        if isinstance(identifier, str):
            return deserialize(identifier, custom_objects=globals())
        else:
            raise ValueError('Could not interpret:',  identifier)
