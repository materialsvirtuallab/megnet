"""
Multiprocessing utilities.
"""

from multiprocessing import Pool
from typing import Iterable, Callable

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = None


def imap_tqdm(nprocs: int, func: Callable, iterable: Iterable, *args, **kwargs):
    """
    A wrapper around Pool.imap. Creates a Pool with nprocs and then runs a f
    unction over an iterable with progress bar.

    :param nprocs: Number of processes
    :param func: Callable
    :param iterable: Iterable of arguments.
    :param args: Passthrough to Pool.imap
    :param kwargs: Passthrough to Pool.imap
    :return: Results of Pool.imap.
    """
    if tqdm is None:
        raise ImportError("tqdm must be installed for this function.")
    data = []
    with Pool(nprocs) as pool:
        try:
            n = len(iterable)  # type: ignore
        except TypeError:
            n = None  # type: ignore
        with tqdm(total=n) as pbar:
            for i, d in enumerate(pool.imap(func, iterable, *args, **kwargs)):
                pbar.update()
                data.append(d)
    return data
