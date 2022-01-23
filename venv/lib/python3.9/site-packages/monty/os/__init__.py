"""
Os functions, e.g., cd, makedirs_p.
"""

import os
import errno

from contextlib import contextmanager

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "1/24/14"


@contextmanager
def cd(path):
    """
    A Fabric-inspired cd context that temporarily changes directory for
    performing some tasks, and returns to the original working directory
    afterwards. E.g.,

        with cd("/my/path/"):
            do_something()

    Args:
        path: Path to cd to.
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def makedirs_p(path, **kwargs):
    """
    Wrapper for os.makedirs that does not raise an exception if the directory
    already exists, in the fashion of "mkdir -p" command. The check is
    performed in a thread-safe way

    Args:
        path: path of the directory to create
        kwargs: standard kwargs for os.makedirs
    """

    try:
        os.makedirs(path, **kwargs)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
