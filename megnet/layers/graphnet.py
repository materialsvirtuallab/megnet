from megnet.layers.base import GraphNetwork
import tensorflow as tf
import keras.backend as K
from megnet.utils.layer_util import repeat_with_index

__author__ = "Chi Chen"
__copyright__ = "Copyright 2018, Materials Virtual Lab "
__version__ = "0.1"
__date__ = "Dec 1, 2018"


class MEGNet(GraphNetwork):
    """
    TODO: Document this?
    """

    def __init__(self,
                 units_v,
                 units_e,
                 units_u,
                 pool_method='mean',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(MEGNet, self).__init__(activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint,
                                     **kwargs)
        self.units_v = units_v
        self.units_e = units_e
        self.units_u = units_u
        self.pool_method = pool_method
        if pool_method == 'mean':
            self.reduce_method = tf.reduce_mean
            self.seg_method = tf.segment_mean
        elif pool_method == 'sum':
            self.reduce_method = tf.reduce_sum
            self.seg_method = tf.segment_sum
        else:
            raise ValueError('Pool method: ' + pool_method + ' not understood!')

    def build(self, input_shapes):
        vdim = input_shapes[0][2]
        edim = input_shapes[1][2]
        udim = input_shapes[2][2]

        with K.name_scope(self.name):
            with K.name_scope('phi_v'):
                v_shapes = [self.units_e[-1] + vdim + udim] + self.units_v
                v_shapes = list(zip(v_shapes[:-1], v_shapes[1:]))
                self.phi_v_weight = [self.add_weight(shape=i,
                                                     initializer=self.kernel_initializer,
                                                     name='weight_v_%d' % j,
                                                     regularizer=self.kernel_regularizer,
                                                     constraint=self.kernel_constraint)
                                     for j, i in enumerate(v_shapes)]
                if self.use_bias:
                    self.phi_v_bias = [self.add_weight(shape=(i[-1],),
                                                       initializer=self.bias_initializer,
                                                       name='bias_v_%d' % j,
                                                       regularizer=self.bias_regularizer,
                                                       constraint=self.bias_constraint)
                                       for j, i in enumerate(v_shapes)]
                else:
                    self.phi_v_bias = None

            with K.name_scope('phi_e'):
                e_shapes = [2 * vdim + edim + udim] + self.units_e
                e_shapes = list(zip(e_shapes[:-1], e_shapes[1:]))
                self.phi_e_weight = [self.add_weight(shape=i,
                                                     initializer=self.kernel_initializer,
                                                     name='weight_e_%d' % j,
                                                     regularizer=self.kernel_regularizer,
                                                     constraint=self.kernel_constraint)
                                     for j, i in enumerate(e_shapes)]
                if self.use_bias:
                    self.phi_e_bias = [self.add_weight(shape=(i[-1],),
                                                       initializer=self.bias_initializer,
                                                       name='bias_e_%d' % j,
                                                       regularizer=self.bias_regularizer,
                                                       constraint=self.bias_constraint)
                                       for j, i in enumerate(e_shapes)]
                else:
                    self.phi_e_bias = None

            with K.name_scope('phi_u'):
                u_shapes = [self.units_e[-1] + self.units_v[
                    -1] + udim] + self.units_u
                u_shapes = list(zip(u_shapes[:-1], u_shapes[1:]))
                self.phi_u_weight = [self.add_weight(shape=i,
                                                     initializer=self.kernel_initializer,
                                                     name='weight_u_%d' % j,
                                                     regularizer=self.kernel_regularizer,
                                                     constraint=self.kernel_constraint)
                                     for j, i in enumerate(u_shapes)]
                if self.use_bias:
                    self.phi_u_bias = [self.add_weight(shape=(i[-1],),
                                                       initializer=self.bias_initializer,
                                                       name='bias_u_%d' % j,
                                                       regularizer=self.bias_regularizer,
                                                       constraint=self.bias_constraint)
                                       for j, i in enumerate(u_shapes)]
                else:
                    self.phi_u_bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        node_feature_shape = input_shape[0]
        edge_feature_shape = input_shape[1]
        graph_feature_shape = input_shape[2]
        output_shape = [
            (node_feature_shape[0], node_feature_shape[1], self.units_v[-1]),
            (edge_feature_shape[0], edge_feature_shape[1], self.units_e[-1]),
            (graph_feature_shape[0], graph_feature_shape[1], self.units_u[-1])]
        return output_shape

    def phi_e(self, inputs):
        node, weights, u, index1, index2, gnode, gbond = inputs
        index1 = tf.reshape(index1, (-1,))
        index2 = tf.reshape(index2, (-1,))
        fs = tf.gather(node, index1, axis=1)
        fr = tf.gather(node, index2, axis=1)
        # print(fs.shape)
        concate_node = tf.concat([fs, fr], axis=-1)
        u_expand = repeat_with_index(u, gbond, axis=1)
        concated = tf.concat([concate_node, weights, u_expand], axis=-1)
        return self._mlp(concated, self.phi_e_weight, self.phi_e_bias)

    def rho_e_v(self, e_p, inputs):
        node, weights, u, index1, index2, gnode, gbond = inputs
        index1 = tf.reshape(index1, (-1,))
        return tf.expand_dims(self.seg_method(tf.squeeze(e_p), index1), axis=0)

    def phi_v(self, b_ei_p, inputs):
        node, weights, u, index1, index2, gnode, gbond = inputs
        u_expand = repeat_with_index(u, gnode, axis=1)
        # print(u_expand.shape, node.shape)
        concated = tf.concat([b_ei_p, node, u_expand], axis=-1)
        return self._mlp(concated, self.phi_v_weight, self.phi_v_bias)

    def rho_e_u(self, e_p, inputs):
        node, weights, u, index1, index2, gnode, gbond = inputs
        gbond = tf.reshape(gbond, (-1,))
        return tf.expand_dims(self.seg_method(tf.squeeze(e_p), gbond), axis=0)

    def rho_v_u(self, v_p, inputs):
        node, weights, u, index1, index2, gnode, gbond = inputs
        gnode = tf.reshape(gnode, (-1,))
        return tf.expand_dims(self.seg_method(tf.squeeze(v_p), gnode), axis=0)

    def phi_u(self, b_e_p, b_v_p, inputs):
        concated = tf.concat([b_e_p, b_v_p, inputs[2]], axis=-1)
        return self._mlp(concated, self.phi_u_weight, self.phi_u_bias)

    def _mlp(self, input_, weights, bias):
        act = input_
        for w, b in zip(weights, bias):
            output = K.dot(act, w) + b
            act = self.activation(output)
        return output

    def get_config(self):
        config = {
            'units_e': self.units_e,
            'units_v': self.units_v,
            'units_u': self.units_u,
            'pool_method': self.pool_method
        }

        base_config = super(MEGNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    import numpy as np

    int32 = 'int32'
    x1 = np.random.rand(1, 5, 10)
    x2 = np.random.rand(1, 6, 5)
    x3 = np.random.rand(1, 2, 20)
    x4 = np.array([0, 1, 2, 3, 3, 4]).reshape([1, -1])
    x5 = np.array([1, 0, 3, 2, 4, 3]).reshape([1, -1])
    x6 = np.array([[0, 0, 1, 1, 1]])
    x7 = np.array([[0, 0, 1, 1, 1, 1]])
    x1_ = Input(shape=(None, 10))
    x2_ = Input(shape=(None, 5))
    x3_ = Input(shape=(None, 20))
    x4_ = Input(shape=(None,), dtype=int32)
    x5_ = Input(shape=(None,), dtype=int32)
    x6_ = Input(shape=(None,), dtype=int32)
    x7_ = Input(shape=(None,), dtype=int32)
    out = MEGNet([10, 5], [20, 4], [30, 3])([x1_, x2_, x3_, x4_, x5_, x6_, x7_])
    model = Model(inputs=[x1_, x2_, x3_, x4_, x5_, x6_, x7_], outputs=out)
    model.compile('adam', 'mse')
    print('Dummy prediction ', model.predict([x1, x2, x3, x4, x5, x6, x7]))
