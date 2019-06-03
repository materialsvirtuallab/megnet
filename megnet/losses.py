import keras.backend as kb


def mean_squared_error_with_scale(y_true, y_pred, scale=10000):
    """
    Keras default log for tracking progress shows two decimal points,
    here we multiply the mse by a factor to fully show the loss in progress bar

    Args:
        y_true: (tensor) training y
        y_pred: (tensor) predicted y
        scale: (int or float) factor to multiply with mse

    Returns:
        scaled mse (float)
    """
    return kb.mean(kb.square(y_pred - y_true), axis=-1) * scale


mse_scale = mean_squared_error_with_scale
