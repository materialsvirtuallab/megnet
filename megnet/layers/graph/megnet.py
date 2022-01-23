"""
Megnet graph layer implementation
"""
import tensorflow as tf
import tensorflow.keras.backend as kb

from megnet.layers.graph import GraphNetworkLayer
from megnet.utils.layer import repeat_with_index

__author__ = "Chi Chen"
__copyright__ = "Copyright 2018, Materials Virtual Lab "
__version__ = "0.1"
__date__ = "Dec 1, 2018"


class MEGNetLayer(GraphNetworkLayer):
    """
    The MEGNet graph implementation as described in the paper

    Chen, Chi; Ye, Weike Ye; Zuo, Yunxing; Zheng, Chen; Ong, Shyue Ping.
    Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals,
    2018, arXiv preprint. [arXiv:1812.05055](https://arxiv.org/abs/1812.05055)
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

    def __init__(
        self,
        units_v,
        units_e,
        units_u,
        pool_method="mean",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        """
        Args:
            units_v (list of integers): the hidden layer sizes for node update neural network
            units_e (list of integers): the hidden layer sizes for edge update neural network
            units_u (list of integers): the hidden layer sizes for state update neural network
            pool_method (str): 'mean' or 'sum', determines how information is gathered to nodes from neighboring edges
            activation (str): Default: None. The activation function used for each sub-neural network. Examples include
                'relu', 'softmax', 'tanh', 'sigmoid' and etc.
            use_bias (bool): Default: True. Whether to use the bias term in the neural network.
            kernel_initializer (str): Default: 'glorot_uniform'. Initialization function for the layer kernel weights,
            bias_initializer (str): Default: 'zeros'
            activity_regularizer (str): Default: None. The regularization function for the output
            kernel_constraint (str): Default: None. Keras constraint for kernel values
            bias_constraint (str): Default: None .Keras constraint for bias values
        """

        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.units_v = units_v
        self.units_e = units_e
        self.units_u = units_u
        self.pool_method = pool_method
        if pool_method == "mean":
            self.reduce_method = tf.reduce_mean
            self.unsorted_seg_method = tf.math.unsorted_segment_mean
            self.seg_method = tf.math.segment_mean
        elif pool_method == "sum":
            self.reduce_method = tf.reduce_sum
            self.seg_method = tf.math.segment_sum
            self.unsorted_seg_method = tf.math.unsorted_segment_sum
        else:
            raise ValueError("Pool method: " + pool_method + " not understood!")

    def build(self, input_shapes):
        """
        Build the weights for the layer
        Args:
            input_shapes (sequence of tuple): the shapes of all input tensors

        """
        vdim = input_shapes[0][2]
        edim = input_shapes[1][2]
        udim = input_shapes[2][2]

        with kb.name_scope(self.name):
            with kb.name_scope("phi_v"):
                v_shapes = [self.units_e[-1] + vdim + udim] + self.units_v
                v_shapes = list(zip(v_shapes[:-1], v_shapes[1:]))
                self.phi_v_weights = [
                    self.add_weight(
                        shape=i,
                        initializer=self.kernel_initializer,
                        name=f"weight_v_{j}",
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                    )
                    for j, i in enumerate(v_shapes)
                ]
                if self.use_bias:
                    self.phi_v_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            initializer=self.bias_initializer,
                            name=f"bias_v_{j}",
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(v_shapes)
                    ]
                else:
                    self.phi_v_biases = None

            with kb.name_scope("phi_e"):
                e_shapes = [2 * vdim + edim + udim] + self.units_e
                e_shapes = list(zip(e_shapes[:-1], e_shapes[1:]))
                self.phi_e_weights = [
                    self.add_weight(
                        shape=i,
                        initializer=self.kernel_initializer,
                        name=f"weight_e_{j}",
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                    )
                    for j, i in enumerate(e_shapes)
                ]
                if self.use_bias:
                    self.phi_e_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            initializer=self.bias_initializer,
                            name=f"bias_e_{j}",
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(e_shapes)
                    ]
                else:
                    self.phi_e_biases = None

            with kb.name_scope("phi_u"):
                u_shapes = [self.units_e[-1] + self.units_v[-1] + udim] + self.units_u
                u_shapes = list(zip(u_shapes[:-1], u_shapes[1:]))
                self.phi_u_weights = [
                    self.add_weight(
                        shape=i,
                        initializer=self.kernel_initializer,
                        name=f"weight_u_{j}",
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                    )
                    for j, i in enumerate(u_shapes)
                ]
                if self.use_bias:
                    self.phi_u_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            initializer=self.bias_initializer,
                            name=f"bias_u_{j}",
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(u_shapes)
                    ]
                else:
                    self.phi_u_biases = None
        self.built = True

    def compute_output_shape(self, input_shape):
        """
        Compute output shapes from input shapes
        Args:
            input_shape (sequence of tuple): input shapes

        Returns: sequence of tuples output shapes

        """
        node_feature_shape = input_shape[0]
        edge_feature_shape = input_shape[1]
        state_feature_shape = input_shape[2]
        output_shape = [
            (node_feature_shape[0], node_feature_shape[1], self.units_v[-1]),
            (edge_feature_shape[0], edge_feature_shape[1], self.units_e[-1]),
            (state_feature_shape[0], state_feature_shape[1], self.units_u[-1]),
        ]
        return output_shape

    def phi_e(self, inputs):
        """
        Edge update function
        Args:
            inputs (tuple of tensor)
        Returns:
            output tensor
        """
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        index1 = tf.reshape(index1, (-1,))
        index2 = tf.reshape(index2, (-1,))
        fs = tf.gather(nodes, index1, axis=1)
        fr = tf.gather(nodes, index2, axis=1)
        concate_node = tf.concat([fs, fr], -1)
        u_expand = repeat_with_index(u, gbond, axis=1)
        concated = tf.concat([concate_node, edges, u_expand], -1)
        return self._mlp(concated, self.phi_e_weights, self.phi_e_biases)

    def rho_e_v(self, e_p, inputs):
        """
        Reduce edge attributes to node attribute, eqn 5 in the paper
        Args:
            e_p: updated bond
            inputs: the whole input list

        Returns: summed tensor

        """
        node, edges, u, index1, index2, gnode, gbond = inputs
        index1 = tf.reshape(index1, (-1,))
        return tf.expand_dims(self.unsorted_seg_method(tf.squeeze(e_p), index1, num_segments=tf.shape(node)[1]), axis=0)

    def phi_v(self, b_ei_p, inputs):
        """
        Node update function
        Args:
            b_ei_p (tensor): edge aggregated tensor
            inputs (tuple of tensors): other graph inputs

        Returns: updated node tensor

        """
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        u_expand = repeat_with_index(u, gnode, axis=1)
        concated = tf.concat([b_ei_p, nodes, u_expand], -1)
        return self._mlp(concated, self.phi_v_weights, self.phi_v_biases)

    def rho_e_u(self, e_p, inputs):
        """
        aggregate edge to state
        Args:
            e_p (tensor): edge tensor
            inputs (tuple of tensors): other graph input tensors

        Returns: edge aggregated tensor for states

        """
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        gbond = tf.reshape(gbond, (-1,))
        return tf.expand_dims(self.seg_method(tf.squeeze(e_p), gbond), axis=0)

    def rho_v_u(self, v_p, inputs):
        """
        Args:
            v_p (tf.Tensor): updated atom/node attributes
            inputs (Sequence): list or tuple for the graph inputs
        Returns:
            atom/node to global/state aggregated tensor
        """
        nodes, edges, u, index1, index2, gnode, gbond = inputs
        gnode = tf.reshape(gnode, (-1,))
        return tf.expand_dims(self.seg_method(tf.squeeze(v_p, axis=0), gnode), axis=0)

    def phi_u(self, b_e_p, b_v_p, inputs):
        """
        Args:
            b_e_p (tf.Tensor): edge/bond to global aggregated tensor
            b_v_p (tf.Tensor): node/atom to global aggregated tensor
            inputs (Sequence): list or tuple for the graph inputs
        Returns:
            updated globa/state attributes
        """
        concated = tf.concat([b_e_p, b_v_p, inputs[2]], -1)
        return self._mlp(concated, self.phi_u_weights, self.phi_u_biases)

    def _mlp(self, input_, weights, biases):
        if biases is None:
            biases = [0] * len(weights)
        act = input_
        for w, b in zip(weights, biases):
            output = kb.dot(act, w) + b
            act = self.activation(output)
        return output

    def get_config(self):
        """
         Part of keras layer interface, where the signature is converted into a dict
        Returns:
            configurational dictionary

        """
        config = {
            "units_e": self.units_e,
            "units_v": self.units_v,
            "units_u": self.units_u,
            "pool_method": self.pool_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
