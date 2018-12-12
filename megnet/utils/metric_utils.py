import numpy as np


def mae(y_true, y_pred):
    """
    Simple mean absolute error calculations
    :param y_true: (numpy array) ground truth
    :param y_pred: (numpy array) predicted values
    :return: (float) mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


def accuracy(y_true, y_pred):
    """
    Simple accuracy calculation
    :param y_true: numpy array of 0 and 1's
    :param y_pred: numpy array of predict sigmoid
    :return: (float) accuracy
    """
    y_pred = y_pred > 0.5
    return np.sum(y_true == y_pred) / len(y_pred)
