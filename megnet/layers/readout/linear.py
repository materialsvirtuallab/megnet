from keras.engine import Layer
import tensorflow as tf


class LinearWithIndex(Layer):
    """
    Sum or average the node/edge attributes to get a structure-level vector

    Args:
        mode: (str) 'mean' or 'sum'
    """
    def __init__(self, mode='mean', **kwargs):
        super(LinearWithIndex, self).__init__(**kwargs)
        self.mode = mode
        if self.mode == 'mean':
            self.reduce_method = tf.math.segment_mean
        elif self.mode == 'sum':
            self.reduce_method = tf.math.segment_sum
        else:
            raise ValueError('Only sum and mean are supported at the moment!')

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
