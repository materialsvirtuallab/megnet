import keras.backend as K


def softplus2(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow

    Args:
        x: (Tensor) input tensor

    Returns:
         (Tensor) output tensor
    """
    return K.relu(x) + K.log(0.5*K.exp(-K.abs(x)) + 0.5)

