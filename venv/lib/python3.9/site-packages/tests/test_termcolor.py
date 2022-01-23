"""
TODO: Modify unittest doc.
"""

import unittest
import os
import sys

from monty.termcolor import (
    cprint,
    cprint_map,
    enable,
    get_terminal_size,
    stream_has_colours,
)


class FuncTest(unittest.TestCase):
    def test_remove_non_ascii(self):
        enable(True)
        print("Current terminal type: %s" % os.getenv("TERM"))
        print("Test basic colors:")
        cprint("Grey color", "grey")
        cprint("Red color", "red")
        cprint("Green color", "green")
        cprint("Yellow color", "yellow")
        cprint("Blue color", "blue")
        cprint("Magenta color", "magenta")
        cprint("Cyan color", "cyan")
        cprint("White color", "white")
        print("-" * 78)

        print("Test highlights:")
        cprint("On grey color", on_color="on_grey")
        cprint("On red color", on_color="on_red")
        cprint("On green color", on_color="on_green")
        cprint("On yellow color", on_color="on_yellow")
        cprint("On blue color", on_color="on_blue")
        cprint("On magenta color", on_color="on_magenta")
        cprint("On cyan color", on_color="on_cyan")
        cprint("On white color", color="grey", on_color="on_white")
        print("-" * 78)

        print("Test attributes:")
        cprint("Bold grey color", "grey", attrs=["bold"])
        cprint("Dark red color", "red", attrs=["dark"])
        cprint("Underline green color", "green", attrs=["underline"])
        cprint("Blink yellow color", "yellow", attrs=["blink"])
        cprint("Reversed blue color", "blue", attrs=["reverse"])
        cprint("Concealed Magenta color", "magenta", attrs=["concealed"])
        cprint(
            "Bold underline reverse cyan color",
            "cyan",
            attrs=["bold", "underline", "reverse"],
        )
        cprint(
            "Dark blink concealed white color",
            "white",
            attrs=["dark", "blink", "concealed"],
        )
        print("-" * 78)

        print("Test mixing:")
        cprint("Underline red on grey color", "red", "on_grey", ["underline"])
        cprint("Reversed green on red color", "green", "on_red", ["reverse"])

        # Test cprint_keys
        cprint_map("Hello world", {"Hello": "red"})
        cprint_map("Hello world", {"Hello": {"color": "blue", "on_color": "on_red"}})

        # Test terminal size.
        print("terminal size: %s", get_terminal_size())
        enable(False)

    def test_stream_has_colors(self):
        # TODO: not a real test. Need to do a proper test.
        stream_has_colours(sys.stdout)


if __name__ == "__main__":
    unittest.main()
