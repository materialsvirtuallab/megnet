"""
Preprocessing codes
"""

from typing import List

import numpy as np
from monty.json import MSONable

from .typing import StructureOrMolecule, VectorLike


class Scaler(MSONable):
    """
    Base Scaler class. It implements transform and
    inverse_transform. Both methods will take number
    of atom as the second parameter in addition to
    the target property
    """

    def transform(self, target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Transform the target values into new target values
        Args:
            target (float): target numerical value
            n (int): number of atoms
        Returns:
            scaled target

        """
        raise NotImplementedError

    def inverse_transform(self, transformed_target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Inverse transform of the target

        Args:
            transformed_target (np.ndarray): transformed target
            n (int): number of atoms

        Returns:
            target
        """
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    Standard scaler with consideration of extensive/intensive quantity
    For intensive quantity, the mean is just the mean of training data,
    and std is the std of training data
    For extensive quantity, the mean is the mean of target/atom, and
    std is the std for target/atom

    Methods:
        transform(self, target, n=1): standard scaling the target and
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, is_intensive: bool = True):
        """

        Args:
            mean (float): mean value of target
            std (float): standard deviation of target
            is_intensive (bool): whether the target is already an intensive
                property
        """
        self.mean = mean
        if np.abs(std) < np.finfo(float).eps:
            std = 1.0
        self.std = std
        self.is_intensive = is_intensive

    def transform(self, target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Transform numeric values according the mean and std, plus a factor n

        Args:
            target (np.ndarray): target numerical value
            n (int): number of atoms
        Returns:
            scaled target
        """
        if self.is_intensive:
            n = 1
        return (target / n - self.mean) / self.std

    def inverse_transform(self, transformed_target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Inverse transform of the target

        Args:
            transformed_target (np.ndarray): transformed target
            n (int): number of atoms

        Returns:
            original target
        """
        if self.is_intensive:
            n = 1
        return n * (transformed_target * self.std + self.mean)

    @classmethod
    def from_training_data(
        cls, structures: List[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True
    ) -> "StandardScaler":
        """
        Generate a target scaler from a list of input structures/molecules,
        a target value vector and an indicator for intensiveness of the
        property

        Args:
            structures (list): list of structures/molecules
            targets (list): vector of target properties
            is_intensive (bool): whether the target is intensive

        Returns: new instance

        """
        if is_intensive:
            new_targets = targets
        else:
            new_targets = [i / len(j) for i, j in zip(targets, structures)]
        mean = np.mean(new_targets).item()
        std = np.std(new_targets).item()
        return cls(mean, std, is_intensive)

    def __str__(self):
        return f"StandardScaler(mean={self.mean:.3f}, std={self.std:.3f}, " f"is_intensive={self.is_intensive})"

    def __repr__(self):
        return str(self)


class DummyScaler(MSONable):
    """
    Dummy scaler does nothing
    """

    @staticmethod
    def transform(target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Args:
            target (np.ndarray): target numerical value
            n (int): number of atoms
        Returns:
            target
        """
        return target

    @staticmethod
    def inverse_transform(transformed_target: np.ndarray, n: int = 1) -> np.ndarray:
        """
        return as it is

        Args:
            transformed_target (np.ndarray): transformed target
            n (int): number of atoms

        Returns:
            transformed_target
        """
        return transformed_target

    @classmethod
    def from_training_data(cls, structures: List[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True):
        """
        Args:
            structures (list): list of structures/molecules
            targets (list): vector of target properties
            is_intensive (bool): whether the target is intensive

        Returns: DummyScaler

        """
        return cls()
