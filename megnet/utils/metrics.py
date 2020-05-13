"""metrics for evaluating datasets"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple mean absolute error calculations

    Args:
        y_true: (numpy array) ground truth
        y_pred: (numpy array) predicted values
    Returns:
         (float) mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred)).item()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple accuracy calculation

    Args:
        y_true: numpy array of 0 and 1's
        y_pred: numpy array of predict sigmoid
    Returns:
        (float) accuracy
    """
    y_pred = y_pred > 0.5
    return np.sum(y_true == y_pred) / len(y_pred)
