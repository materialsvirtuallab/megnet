"""
Additional tools for iteration.
"""
import itertools

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def chunks(items, n):
    """
    Yield successive n-sized chunks from a list-like object.

    >>> import pprint
    >>> pprint.pprint(list(chunks(range(1, 25), 10)))
    [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
     (11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
     (21, 22, 23, 24)]
    """
    it = iter(items)
    chunk = tuple(itertools.islice(it, n))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, n))


def iterator_from_slice(s):
    """
    Constructs an iterator given a slice object s.

    .. note::

        The function returns an infinite iterator if s.stop is None
    """
    start = s.start if s.start is not None else 0
    step = s.step if s.step is not None else 1

    if s.stop is None:
        # Infinite iterator.
        return itertools.count(start=start, step=step)

    # xrange-like iterator that supports float.
    return iter(np.arange(start, s.stop, step))


def iuptri(items, diago=True, with_inds=False):
    """
    A generator that yields the upper triangle of the matrix (items x items)

    Args:
        items: Iterable object with elements [e0, e1, ...]
        diago: False if diagonal matrix elements should be excluded
        with_inds: If True, (i,j) (e_i, e_j) is returned else (e_i, e_j)

    >>> for (ij, mate) in iuptri([0,1], with_inds=True):
    ...     print("ij:", ij, "mate:", mate)
    ij: (0, 0) mate: (0, 0)
    ij: (0, 1) mate: (0, 1)
    ij: (1, 1) mate: (1, 1)
    """
    for (ii, item1) in enumerate(items):
        for (jj, item2) in enumerate(items):
            do_yield = (jj >= ii) if diago else (jj > ii)
            if do_yield:
                if with_inds:
                    yield (ii, jj), (item1, item2)
                else:
                    yield item1, item2


def ilotri(items, diago=True, with_inds=False):
    """
    A generator that yields the lower triangle of the matrix (items x items)

    Args:
        items: Iterable object with elements [e0, e1, ...]
        diago: False if diagonal matrix elements should be excluded
        with_inds: If True, (i,j) (e_i, e_j) is returned else (e_i, e_j)

    >>> for (ij, mate) in ilotri([0,1], with_inds=True):
    ...     print("ij:", ij, "mate:", mate)
    ij: (0, 0) mate: (0, 0)
    ij: (1, 0) mate: (1, 0)
    ij: (1, 1) mate: (1, 1)
    """
    for (ii, item1) in enumerate(items):
        for (jj, item2) in enumerate(items):
            do_yield = (jj <= ii) if diago else (jj < ii)
            if do_yield:
                if with_inds:
                    yield (ii, jj), (item1, item2)
                else:
                    yield item1, item2
