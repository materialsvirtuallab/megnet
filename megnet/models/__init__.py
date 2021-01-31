"""
Models package, this package contains various graph-based models
"""
from .base import GraphModel
from .megnet import MEGNetModel

__all__ = ["GraphModel", "MEGNetModel"]
