import numpy as np
from monty.json import MSONable


class StandardScaler(MSONable):
    """
    Standard scaler with consideration of extensive/intensive quantity
    For intensive quantity, the mean is just the mean of training data, and std
    is the std of training data
    For extensive quantity, the mean is the mean of target/atom, and std is the
    std for target/atom

    Args:
        mean (float): mean value of target
        std (float): standard deviation of target
        is_intensive (bool): whether the target is already an intensive property

    Methods:
        transform(self, target, n=1): standard scaling the target and
    """

    def __init__(self, mean=0, std=1, is_intensive=True):
        self.mean = mean
        if std == 0:
            std = 1
        self.std = std
        self.is_intensive = is_intensive

    def transform(self, target, n=1):
        """
        Transform numeric values according the mean and std, plus a factor n

        Args:
            target: target numerical value
            n: number of atoms
        Returns:
            scaled target
        """
        if self.is_intensive:
            n = 1
        return (target / n - self.mean) / self.std

    def inverse_transform(self, transformed_target, n=1):
        """
        Inverse transform of the target

        Args:
            transformed_target: transformed target
            n: number of atoms

        Returns:
            original target
        """
        if self.is_intensive:
            n = 1
        return n * (transformed_target * self.std + self.mean)

    @classmethod
    def from_training_data(cls, structures, targets, is_intensive=True):
        if is_intensive:
            new_targets = targets
        else:
            new_targets = [i / len(j) for i, j in zip(targets, structures)]
        mean = np.mean(new_targets)
        std = np.std(new_targets)
        return cls(mean, std, is_intensive)

    def __str__(self):
        return "StandardScaler(mean=%.3f, std=%.3f, is_intensive=%d)" % (self.mean, self.std, self.is_intensive)

    def __repr__(self):
        return str(self)


class DummyScaler(MSONable):
    """
    Dummy scaler does nothing
    """

    def transform(self, target, n=1):
        return target

    def inverse_transform(self, transformed_target, n=1):
        return transformed_target

    @classmethod
    def from_training_data(cls, structures, targets, is_intensive=True):
        return cls()
