"""
Addition math functions.
"""

import math


def nCr(n, r):
    """
    Calculates nCr.

    Args:
        n (int): total number of items.
        r (int): items to choose

    Returns:
        nCr.
    """
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))


def nPr(n, r):
    """
    Calculates nPr.

    Args:
        n (int): total number of items.
        r (int): items to permute

    Returns:
        nPr.
    """
    f = math.factorial
    return int(f(n) / f(n - r))
