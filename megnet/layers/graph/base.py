"""
A full GN block has the following computation steps
 1. Compute updated edge attributes
 2. Aggregate edge attributes per node
 3. Compute updated node attributes
 4. Aggregate edge attributes globally
 5. Aggregate node attributes globally
 6. Compute updated global attribute

[1] https://arxiv.org/pdf/1806.01261.pdf
"""

from keras.engine import Layer
from keras import regularizers, constraints, initializers
from megnet import activations


class GraphNetworkLayer(Layer):
    """
    Implementation of a graph network layer. Current implementation is based on
    neural networks for each update function, and sum or mean for each
    aggregation function

    Args:
        activation (str): Default: None. The activation function used for each
            sub-neural network. Examples include 'relu', 'softmax', 'tanh',
            'sigmoid' and etc.
        use_bias (bool): Default: True. Whether to use the bias term in the
            neural network.
        kernel_initializer (str): Default: 'glorot_uniform'. Initialization
            function for the layer kernel weights,
        bias_initializer (str): Default: 'zeros'
        activity_regularizer (str): Default: None. The regularization function
            for the output
        kernel_constraint (str): Default: None. Keras constraint for kernel
            values
        bias_constraint (str): Default: None .Keras constraint for bias values

    Method:
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
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        e_p = self.phi_e(inputs)
        b_ei_p = self.rho_e_v(e_p, inputs)
        v_p = self.phi_v(b_ei_p, inputs)
        b_e_p = self.rho_e_u(e_p, inputs)
        b_v_p = self.rho_v_u(v_p, inputs)
        u_p = self.phi_u(b_e_p, b_v_p, inputs)
        return [v_p, e_p, u_p]

    def compute_output_shape(self, input_shape):
        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def phi_e(self, inputs):
        """
        This is for updating the edge attributes
        ek' = phi_e(ek, vrk, vsk, u)
        :return:
        """
        raise NotImplementedError

    def rho_e_v(self, e_p, inputs):
        """
        This is for step 2, aggregate edge attributes per node
        Ei' = {(ek', rk, sk)} with rk =i, k=1:Ne

        \bar e_i' = rho_e_v(Ei')
        :return:
        """
        raise NotImplementedError

    def phi_v(self, b_e_p, inputs):
        """
        Step 3. Compute updated node attributes
        v_i' = phi_v(\bar e_i, vi, u)
        :return:
        """
        raise NotImplementedError

    def rho_e_u(self, e_p, inputs):
        """
        let V' = {v'} i = 1:Nv
        let E' = {(e_k', rk, sk)} k = 1:Ne
        \bar e' = rho_e_u(E')
        :return:
        """
        raise NotImplementedError

    def rho_v_u(self, v_p, inputs):
        """
        \bar v' = rho_v_u(V')

        :return:
        """
        raise NotImplementedError

    def phi_u(self, b_e_p, b_v_p, inputs):
        """
        u' = phi_u(\bar e', \bar v', u)
        :return:
        """
        raise NotImplementedError

    def get_config(self):
        """
        Part of keras layer interface, where the signature is converted into a dict
        :return:
        """
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(
                self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(
                self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(
                self.activity_regularizer),
            'kernel_constraint': constraints.serialize(
                self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
