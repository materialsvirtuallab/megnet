"""
Math functions.
"""
from math import gcd as pygcd


def gcd(*numbers):
    r"""
    Returns the greatest common divisor for a sequence of numbers.

    Args:
        *numbers: Sequence of numbers.

    Returns:
        (int) Greatest common divisor of numbers.
    """
    n = numbers[0]
    for i in numbers:
        n = pygcd(n, i)
    return n


def lcm(*numbers):
    r"""
    Return lowest common multiple of a sequence of numbers.

    Args:
        *numbers: Sequence of numbers.

    Returns:
        (int) Lowest common multiple of numbers.
    """
    n = 1
    for i in numbers:
        n = (i * n) // gcd(i, n)
    return n


def gcd_float(numbers, tol=1e-8):
    """
    Returns the greatest common divisor for a sequence of numbers.
    Uses a numerical tolerance, so can be used on floats

    Args:
        numbers: Sequence of numbers.
        tol: Numerical tolerance

    Returns:
        (int) Greatest common divisor of numbers.
    """

    def pair_gcd_tol(a, b):
        """Calculate the Greatest Common Divisor of a and b.

        Unless b==0, the result will have the same sign as b (so that when
        b is divided by it, the result comes out positive).
        """
        while b > tol:
            a, b = b, a % b
        return a

    n = numbers[0]
    for i in numbers:
        n = pair_gcd_tol(n, i)
    return n
