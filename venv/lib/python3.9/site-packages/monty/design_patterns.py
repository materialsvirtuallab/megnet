"""
Some common design patterns such as singleton and cached classes.
"""

import os
from functools import wraps


def singleton(cls):
    """
    This decorator can be used to create a singleton out of a class.

    Usage::

        @singleton
        class MySingleton():

            def __init__():
                pass
    """

    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


def cached_class(klass):
    """
    Decorator to cache class instances by constructor arguments.
    This results in a class that behaves like a singleton for each
    set of constructor arguments, ensuring efficiency.

    Note that this should be used for *immutable classes only*.  Having
    a cached mutable class makes very little sense.  For efficiency,
    avoid using this decorator for situations where there are many
    constructor arguments permutations.

    The keywords argument dictionary is converted to a tuple because
    dicts are mutable; keywords themselves are strings and
    so are always hashable, but if any arguments (keyword
    or positional) are non-hashable, that set of arguments
    is not cached.
    """
    cache = {}

    @wraps(klass, assigned=("__name__", "__module__"), updated=())
    class _decorated(klass):
        # The wraps decorator can't do this because __doc__
        # isn't writable once the class is created
        __doc__ = klass.__doc__

        def __new__(cls, *args, **kwargs):
            """
            Pass through...
            :param args:
            :param kwargs:
            :return:
            """
            key = (cls,) + args + tuple(kwargs.items())
            try:
                inst = cache.get(key, None)
            except TypeError:
                # Can't cache this set of arguments
                inst = key = None
            if inst is None:
                # Technically this is cheating, but it works,
                # and takes care of initializing the instance
                # (so we can override __init__ below safely);
                # calling up to klass.__new__ would be the
                # "official" way to create the instance, but
                # that raises DeprecationWarning if there are
                # args or kwargs and klass does not override
                # __new__ (which most classes don't), because
                # object.__new__ takes no parameters (and in
                # Python 3 the warning will become an error)
                inst = klass(*args, **kwargs)
                # This makes isinstance and issubclass work
                # properly
                inst.__class__ = cls
                if key is not None:
                    cache[key] = inst
            return inst

        def __init__(self, *args, **kwargs):
            # This will be called every time __new__ is
            # called, so we skip initializing here and do
            # it only when the instance is created above
            pass

    return _decorated


class NullFile:
    """A file object that is associated to /dev/null."""

    def __new__(cls):
        """
        Pass through
        """
        return open(os.devnull, "w")  # pylint: disable=R1732

    def __init__(self):
        """no-op"""


class NullStream:
    """A fake stream with a no-op write.."""

    def write(*args):  # pylint: disable=E0211
        """
        Does nothing...
        :param args:
        """
        pass
