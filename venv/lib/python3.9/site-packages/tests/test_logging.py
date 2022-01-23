import unittest
from io import StringIO

import logging
from monty.logging import logged


@logged()
def add(a, b):
    return a + b


class FuncTest(unittest.TestCase):
    def test_logged(self):
        s = StringIO()
        logging.basicConfig(level=logging.DEBUG, stream=s)
        add(1, 2)


if __name__ == "__main__":
    unittest.main()
