"""
Gaussian expansion of distances
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class GaussianExpansion(Layer):
    """
    Simple Gaussian expansion.
    A vector of distance [d1, d2, d3, ..., dn] is expanded to a
    matrix of shape [n, m], where m is the number of Gaussian basis centers

    """

    def __init__(self, centers, width, **kwargs):
        """
        Args:
            centers (np.ndarray): Gaussian basis centers
            width (float): width of the Gaussian basis
            **kwargs:
        """
        self.centers = np.array(centers).ravel()
        self.width = width
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        build the layer
        Args:
            input_shape (tuple): tuple of int for the input shape
        """
        self.built = True

    def call(self, inputs, masks=None):
        """
        The core logic function

        Args:
            inputs (tf.Tensor): input distance tensor, with shape [None, n]
            masks (tf.Tensor): bool tensor, not used here
        """
        return tf.math.exp(-((inputs[:, :, None] - self.centers[None, None, :]) ** 2) / self.width**2)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape, used in older keras API
        """
        return input_shape[0], input_shape[1], len(self.centers)

    def get_config(self):
        """
        Get layer configurations
        """
        base_config = super().get_config()
        config = {"centers": self.centers.tolist(), "width": self.width}
        return dict(list(base_config.items()) + list(config.items()))
