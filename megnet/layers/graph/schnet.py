from megnet.layers.graph.base import GraphNetworkLayer
from megnet.activations import softplus2
import keras.backend as kb
import tensorflow as tf


class InteractionLayer(GraphNetworkLayer):
    """
    The Continuous filter InteractionLayer in Schnet

    Schütt et al. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions

    Args:
        activation (str): Default: None. The activation function used for each sub-neural network. Examples include
            'relu', 'softmax', 'tanh', 'sigmoid' and etc.
        use_bias (bool): Default: True. Whether to use the bias term in the neural network.
        kernel_initializer (str): Default: 'glorot_uniform'. Initialization function for the layer kernel weights,
        bias_initializer (str): Default: 'zeros'
        activity_regularizer (str): Default: None. The regularization function for the output
        kernel_constraint (str): Default: None. Keras constraint for kernel values
        bias_constraint (str): Default: None .Keras constraint for bias values

    Methods:
        call(inputs, mask=None): the logic of the layer, returns the final graph
        compute_output_shape(input_shape): compute static output shapes, returns list of tuple shapes
        build(input_shape): initialize the weights and biases for each function
        phi_e(inputs): update function for bonds and returns updated bond attribute e_p
        rho_e_v(e_p, inputs): aggregate updated bonds e_p to per atom attributes, b_e_p
        phi_v(b_e_p, inputs): update the atom attributes by the results from previous step b_e_p and all the inputs
            returns v_p.
        rho_e_u(e_p, inputs): aggregate bonds to global attribute
        rho_v_u(v_p, inputs): aggregate atom to global attributes
        get_config(): part of keras interface for serialization

    """

    def __init__(self,
                 activation=softplus2,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)

    def build(self, input_shapes):
        vdim = input_shapes[0][2]
        edim = input_shapes[1][2]

        with kb.name_scope(self.name):
            with kb.name_scope('phi_e'):
                e_shapes = [[edim, vdim]] + [[vdim, vdim]] * 2
                self.phi_e_weights = [self.add_weight(shape=i,
                                                      initializer=self.kernel_initializer,
                                                      name='weight_v_%d' % j,
                                                      regularizer=self.kernel_regularizer,
                                                      constraint=self.kernel_constraint)
                                      for j, i in enumerate(e_shapes)]
                if self.use_bias:
                    self.phi_e_biases = [self.add_weight(shape=(i[-1],),
                                                         initializer=self.bias_initializer,
                                                         name='bias_v_%d' % j,
                                                         regularizer=self.bias_regularizer,
                                                         constraint=self.bias_constraint)
                                         for j, i in enumerate(e_shapes)]
                else:
                    self.phi_e_biases = None

        with kb.name_scope(self.name):
            with kb.name_scope('phi_v'):

                v_shapes = [[vdim, vdim]] + [[vdim, vdim]] * 2
                self.phi_v_weights = [self.add_weight(shape=i,
                                                      initializer=self.kernel_initializer,
                                                      name='weight_v_%d' % j,
                                                      regularizer=self.kernel_regularizer,
                                                      constraint=self.kernel_constraint)
                                      for j, i in enumerate(v_shapes)]
                if self.use_bias:
                    self.phi_v_biases = [self.add_weight(shape=(i[-1],),
                                                         initializer=self.bias_initializer,
                                                         name='bias_v_%d' % j,
                                                         regularizer=self.bias_regularizer,
                                                         constraint=self.bias_constraint)
                                         for j, i in enumerate(v_shapes)]
                else:
                    self.phi_v_biases = None
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def phi_e(self, inputs):
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        return edges

    def rho_e_v(self, e_p, inputs):
        """
        Reduce edge attributes to node attribute, eqn 5 in the paper
        Args:
            e_p: updated bond
            inputs: the whole input list

        Returns: summed tensor

        """
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        atomwise1 = self._mlp(nodes, self.phi_v_weights[0], self.phi_v_biases[0])

        cfconv1 = self.activation(self._mlp(edges, self.phi_e_weights[0], self.phi_e_biases[0]))
        cfconv2 = self.activation(self._mlp(cfconv1, self.phi_e_weights[1], self.phi_e_biases[1]))
        cfconv_out = self._mlp(cfconv2, self.phi_e_weights[2], self.phi_e_biases[2])

        index1 = tf.reshape(index1, (-1,))
        index2 = tf.reshape(index2, (-1,))
        fr = tf.gather(atomwise1, index2, axis=1)

        after_cfconv = atomwise1 + \
            tf.transpose(a=tf.math.segment_sum(tf.transpose(a=fr * cfconv_out, perm=[1, 0, 2]), index1), perm=[1, 0, 2])

        atomwise2 = self.activation(self._mlp(after_cfconv, self.phi_v_weights[1], self.phi_v_biases[1]))
        atomwise3 = self._mlp(atomwise2, self.phi_v_weights[2], self.phi_v_biases[2])
        return atomwise3

    def phi_v(self, b_ei_p, inputs):
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        return nodes + b_ei_p

    def rho_e_u(self, e_p, inputs):
        return 0

    def rho_v_u(self, v_p, inputs):
        return 0

    def phi_u(self, b_e_p, b_v_p, inputs):
        return inputs[2]

    def _mlp(self, input_, weights, bias):
        output = kb.dot(input_, weights) + bias
        return output

    def get_config(self):
        base_config = super().get_config()
        return base_config
