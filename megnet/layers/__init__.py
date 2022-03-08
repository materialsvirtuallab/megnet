"""Megnet layers implementations.
This subpackage includes

1. Graph convolution layers
2. Readout layers
"""

from tensorflow.keras.layers import deserialize as keras_layer_deserialize

from megnet.activations import softplus2, swish
from megnet.layers.featurizer import GaussianExpansion
from megnet.layers.graph import CrystalGraphLayer, InteractionLayer, MEGNetLayer
from megnet.layers.readout import LinearWithIndex, Set2Set
from megnet.losses import mean_squared_error_with_scale

_CUSTOM_OBJECTS = globals()


__all__ = [
    "MEGNetLayer",
    "CrystalGraphLayer",
    "InteractionLayer",
    "Set2Set",
    "LinearWithIndex",
    "GaussianExpansion",
    "keras_layer_deserialize",
    "mean_squared_error_with_scale",
    "softplus2",
    "swish",
]
