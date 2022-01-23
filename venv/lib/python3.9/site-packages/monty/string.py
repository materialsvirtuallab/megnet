"""
Useful additional string functions.
"""

import sys


def remove_non_ascii(s):
    """
    Remove non-ascii characters in a file. Needed when support for non-ASCII
    is not available.

    Args:
        s (str): Input string

    Returns:
        String with all non-ascii characters removed.
    """
    return "".join(i for i in s if ord(i) < 128)


def unicode2str(s):
    """
    Forces a unicode to a string in Python 2, but transparently handles
    Python 3.

    Args:
        s (str/unicode): Input string / unicode.

    Returns:
        str in Python 2. Unchanged otherwise.
    """
    return s.encode("utf-8") if sys.version_info.major < 3 else s


def is_string(s):
    """True if s behaves like a string (duck typing test)."""
    try:
        s + " "
        return True

    except TypeError:
        return False


def list_strings(arg):
    """
    Always return a list of strings, given a string or list of strings as
    input.

    :Examples:

    >>> list_strings('A single string')
    ['A single string']

    >>> list_strings(['A single string in a list'])
    ['A single string in a list']

    >>> list_strings(['A','list','of','strings'])
    ['A', 'list', 'of', 'strings']
    """
    if is_string(arg):
        return [arg]

    return arg


def marquee(text="", width=78, mark="*"):
    """
    Return the input string centered in a 'marquee'.

    Args:
        text (str): Input string
        width (int): Width of final output string.
        mark (str): Character used to fill string.

    :Examples:

    >>> marquee('A test', width=40)
    '**************** A test ****************'

    >>> marquee('A test', width=40, mark='-')
    '---------------- A test ----------------'

    marquee('A test',40, ' ')
    '                 A test                 '
    """
    if not text:
        return (mark * width)[:width]

    nmark = (width - len(text) - 2) // len(mark) // 2
    nmark = max(nmark, 0)

    marks = mark * nmark
    return f"{marks} {text} {marks}"


def boxed(msg, ch="=", pad=5):
    """
    Returns a string in a box

    Args:
        msg: Input string.
        ch: Character used to form the box.
        pad: Number of characters ch added before and after msg.

    >>> print(boxed("hello", ch="*", pad=2))
    ***********
    ** hello **
    ***********
    """
    if pad > 0:
        msg = pad * ch + " " + msg.strip() + " " + pad * ch

    return "\n".join(
        [
            len(msg) * ch,
            msg,
            len(msg) * ch,
        ]
    )


def make_banner(s, width=78, mark="*"):
    """
    :param s: String
    :param width: Width of banner. Defaults to 78.
    :param mark: The mark used to create the banner.
    :return: Banner string.
    """
    banner = marquee(s, width=width, mark=mark)
    return "\n" + len(banner) * mark + "\n" + banner + "\n" + len(banner) * mark


def indent(lines, amount, ch=" "):
    """
    Indent the lines in a string by padding each one with proper number of pad
    characters
    """
    padding = amount * ch
    return padding + ("\n" + padding).join(lines.split("\n"))
