"""
This module implements several useful functions and decorators that can be
particularly useful for developers. E.g., deprecating methods / classes, etc.
"""

import re
import sys
import logging
import warnings
import os
import subprocess
import multiprocessing
import functools

logger = logging.getLogger(__name__)


def deprecated(replacement=None, message=None, category=FutureWarning):
    """
    Decorator to mark classes or functions as deprecated,
    with a possible replacement.

    Args:
        replacement (callable): A replacement class or method.
        message (str): A warning message to be displayed.
        category (Warning): Choose the category of the warning to issue. Defaults
            to FutureWarning. Another choice can be DeprecationWarning. NOte that
            FutureWarning is meant for end users and is always shown unless silenced.
            DeprecationWarning is meant for developers and is never shown unless
            python is run in developmental mode or the filter is changed. Make
            the choice accordingly.

    Returns:
        Original function, but with a warning to use the updated class.
    """

    def wrap(old):
        def wrapped(*args, **kwargs):
            msg = f"{old.__name__} is deprecated"
            if replacement is not None:
                if isinstance(replacement, property):
                    r = replacement.fget
                elif isinstance(replacement, (classmethod, staticmethod)):
                    r = replacement.__func__
                else:
                    r = replacement
                msg += f"; use {r.__name__} in {r.__module__} instead."
            if message is not None:
                msg += "\n" + message
            warnings.warn(msg, category=category, stacklevel=2)
            return old(*args, **kwargs)

        return wrapped

    return wrap


class requires:
    """
    Decorator to mark classes or functions as requiring a specified condition
    to be true. This can be used to present useful error messages for
    optional dependencies. For example, decorating the following code will
    check if scipy is present and if not, a runtime error will be raised if
    someone attempts to call the use_scipy function::

        try:
            import scipy
        except ImportError:
            scipy = None

        @requires(scipy is not None, "scipy is not present.")
        def use_scipy():
            print(scipy.majver)

    Args:
        condition: Condition necessary to use the class or function.
        message: A message to be displayed if the condition is not True.
    """

    def __init__(self, condition, message):
        """
        :param condition: A expression returning a bool.
        :param message: Message to display if condition is False.
        """
        self.condition = condition
        self.message = message

    def __call__(self, _callable):
        """
        :param _callable: Callable function.
        """

        @functools.wraps(_callable)
        def decorated(*args, **kwargs):
            if not self.condition:
                raise RuntimeError(self.message)
            return _callable(*args, **kwargs)

        return decorated


def get_ncpus():
    """
    .. note::

        If you are using Python >= 2.7, multiprocessing.cpu_count() already
        provides the number of CPUs. In fact, this is the first method tried.
        The purpose of this function is to cater to old Python versions that
        still exist on many Linux style clusters.

    Number of virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program. Return -1 if ncpus cannot be detected. Taken from:
    http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-
    cpus-in-python
    """
    # Python 2.6+
    # May raise NonImplementedError
    try:
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # POSIX
    try:
        res = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ["NUMBER_OF_PROCESSORS"])
        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime  # pylint: disable=import-outside-toplevel

        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        with subprocess.Popen(["sysctl", "-n", "hw.ncpu"], stdout=subprocess.PIPE) as sysctl:
            scstdout = sysctl.communicate()[0]
            res = int(scstdout)
            if res > 0:
                return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open("/proc/cpuinfo").read().count("processor\t:")  # pylint: disable=R1732
        if res > 0:
            return res
    except OSError:
        pass

    # Solaris
    try:
        pseudo_devices = os.listdir("/devices/pseudo/")
        expr = re.compile("^cpuid@[0-9]+$")
        res = 0
        for pd in pseudo_devices:
            if expr.match(pd) is not None:
                res += 1
        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            with open("/var/run/dmesg.boot") as f:
                dmesg = f.read()
        except OSError:
            with subprocess.Popen(["dmesg"], stdout=subprocess.PIPE) as dmesg_process:
                dmesg = dmesg_process.communicate()[0]

        res = 0
        while "\ncpu" + str(res) + ":" in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    logger.warning("Cannot determine number of CPUs on this system!")
    return -1


def install_excepthook(hook_type="color", **kwargs):
    """
    This function replaces the original python traceback with an improved
    version from Ipython. Use `color` for colourful traceback formatting,
    `verbose` for Ka-Ping Yee's "cgitb.py" version kwargs are the keyword
    arguments passed to the constructor. See IPython.core.ultratb.py for more
    info.

    Return:
        0 if hook is installed successfully.
    """
    try:
        from IPython.core import ultratb  # pylint: disable=import-outside-toplevel
    except ImportError:
        warnings.warn("Cannot install excepthook, IPyhon.core.ultratb not available")
        return 1

    # Select the hook.
    hook = dict(
        color=ultratb.ColorTB,
        verbose=ultratb.VerboseTB,
    ).get(hook_type.lower(), None)

    if hook is None:
        return 2

    sys.excepthook = hook(**kwargs)
    return 0
