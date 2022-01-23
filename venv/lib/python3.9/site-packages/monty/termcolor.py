"""
Copyright (c) 2008-2011 Volvox Development Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Konstantin Lepa <konstantin.lepa@gmail.com>

ANSII Color formatting for output in terminal.
"""
import os

try:
    import fcntl
    import termios
    import struct
    import curses
except Exception:
    pass

__all__ = ["colored", "cprint"]

VERSION = (1, 1, 0)

ATTRIBUTES = dict(bold=1, dark=2, underline=4, blink=5, reverse=7, concealed=8)

HIGHLIGHTS = dict(
    on_grey=40,
    on_red=41,
    on_green=42,
    on_yellow=43,
    on_blue=44,
    on_magenta=45,
    on_cyan=46,
    on_white=47,
)

COLORS = dict(grey=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37)

RESET = "\033[0m"

__ISON = True


def enable(true_false):
    """Enable/Disable ANSII Color formatting"""
    global __ISON
    __ISON = true_false


def ison():
    """True if ANSII Color formatting is activated."""
    return __ISON


def stream_has_colours(stream):
    """
    True if stream supports colours. Python cookbook, #475186
    """
    if not hasattr(stream, "isatty"):
        return False

    if not stream.isatty():
        return False  # auto color only on TTYs
    try:
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except Exception:
        return False  # guess false in case of error


def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """

    if __ISON and os.getenv("ANSI_COLORS_DISABLED") is None:
        fmt_str = "\033[%dm%s"
        if color is not None:
            text = fmt_str % (COLORS[color], text)

        if on_color is not None:
            text = fmt_str % (HIGHLIGHTS[on_color], text)

        if attrs is not None:
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)

        text += RESET
    return text


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.

    It accepts arguments of print function.
    """
    try:
        print((colored(text, color, on_color, attrs)), **kwargs)
    except TypeError:
        # flush is not supported by py2.7
        kwargs.pop("flush", None)
        print((colored(text, color, on_color, attrs)), **kwargs)


def colored_map(text, cmap):
    """
    Return colorized text. cmap is a dict mapping tokens to color options.

    .. Example:

        colored_key("foo bar", {bar: "green"})
        colored_key("foo bar", {bar: {"color": "green", "on_color": "on_red"}})
    """
    if not __ISON:
        return text
    for key, v in cmap.items():
        if isinstance(v, dict):
            text = text.replace(key, colored(key, **v))
        else:
            text = text.replace(key, colored(key, color=v))
    return text


def cprint_map(text, cmap, **kwargs):
    """
    Print colorize text.
    cmap is a dict mapping keys to color options.
    kwargs are passed to print function

    Example:
        cprint_map("Hello world", {"Hello": "red"})
    """
    try:
        print(colored_map(text, cmap), **kwargs)
    except TypeError:
        # flush is not supported by py2.7
        kwargs.pop("flush", None)
        print(colored_map(text, cmap), **kwargs)


def get_terminal_size():
    """
    Return the size of the terminal as (nrow, ncols)

    Based on:

        http://stackoverflow.com/questions/566746/how-to-get-console-window-
        width-in-python
    """
    try:
        rc = os.popen("stty size", "r").read().split()
        return int(rc[0]), int(rc[1])
    except Exception:
        pass

    env = os.environ

    def ioctl_GWINSZ(fd):
        try:

            rc = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))
            return rc
        except Exception:
            return None

    rc = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)

    if not rc:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            rc = ioctl_GWINSZ(fd)
            os.close(fd)
        except Exception:
            pass

    if not rc:
        rc = (env.get("LINES", 25), env.get("COLUMNS", 80))

    return int(rc[0]), int(rc[1])
