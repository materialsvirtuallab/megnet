"""
Additional bisect functions. Taken from
https://docs.python.org/2/library/bisect.html
The above bisect() functions are useful for finding insertion points but can be
tricky or awkward to use for common searching tasks.
The functions show how to transform them into the standard lookups for sorted
lists.
"""

import bisect as bs

__author__ = "Matteo Giantomassi"
__copyright__ = "Copyright 2013, The Materials Virtual Lab"
__version__ = "0.1"
__maintainer__ = "Matteo Giantomass"
__email__ = "gmatteo@gmail.com"
__date__ = "11/09/14"


def index(a, x, atol=None):
    """Locate the leftmost value exactly equal to x."""
    i = bs.bisect_left(a, x)
    if i != len(a):
        if atol is None:
            if a[i] == x:
                return i
        else:
            if abs(a[i] - x) < atol:
                return i
    raise ValueError


def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bs.bisect_left(a, x)
    if i:
        return i - 1
    raise ValueError


def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bs.bisect_right(a, x)
    if i:
        return i - 1
    raise ValueError


def find_gt(a, x):
    """Find leftmost value greater than x."""
    i = bs.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError


def find_ge(a, x):
    """Find leftmost item greater than or equal to x."""
    i = bs.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError
