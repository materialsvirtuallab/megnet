import keras.backend as K


def mean_squared_error_with_scale(y_true, y_pred, scale=10000):
    """
    Keras default log for tracking progress shows two decimal points,
    here we multiply the mse by a factor to fully show the loss in progress bar

    :param y_true: (tensor) training y
    :param y_pred: (tensor) predicted y
    :param scale: (int or float) factor to multiply with mse
    :return: scaled mse
    """
    return K.mean(K.square(y_pred - y_true), axis=-1) * scale


mse_scale = mean_squared_error_with_scale
