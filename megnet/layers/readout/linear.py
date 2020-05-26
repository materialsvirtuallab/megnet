from tensorflow.keras.layers import Layer

import tensorflow as tf

MAPPING = {'mean': tf.math.segment_mean,
           'sum': tf.math.segment_sum,
           'max': tf.math.segment_max,
           'min': tf.math.segment_min,
           'prod': tf.math.segment_prod}


class LinearWithIndex(Layer):
    """
    Sum or average the node/edge attributes to get a structure-level vector

    Args:
        mode: (str) 'mean', 'sum', 'max', 'mean' or 'prod'
    """
    def __init__(self, mode='mean', **kwargs):
        super(LinearWithIndex, self).__init__(**kwargs)
        self.mode = mode
        self.reduce_method = MAPPING.get(mode, None)
        if self.reduce_method is None:
            raise ValueError('mode not supported')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, mask=None):
        prop, index = inputs
        index = tf.reshape(index, (-1,))
        prop = tf.transpose(a=prop, perm=[1, 0, 2])
        out = self.reduce_method(prop, index)
        out = tf.transpose(a=out, perm=[1, 0, 2])
        return out

    def compute_output_shape(self, input_shape):
        prop_shape = input_shape[0]
        return prop_shape[0], None, prop_shape[-1]

    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(LinearWithIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
