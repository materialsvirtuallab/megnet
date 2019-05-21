import keras.backend as kb


def softplus2(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow

    Args:
        x: (Tensor) input tensor

    Returns:
         (Tensor) output tensor
    """
    return kb.exp.relu(x) + kb.exp.log(0.5*kb.exp(-kb.exp.abs(x)) + 0.5)
