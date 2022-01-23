"""
TODO: Modify unittest doc.
"""

import unittest
import random
import sys

from monty.string import remove_non_ascii, unicode2str


class FuncTest(unittest.TestCase):
    def test_remove_non_ascii(self):
        s = "".join(chr(random.randint(0, 127)) for i in range(10))
        s += "".join(chr(random.randint(128, 150)) for i in range(10))
        clean = remove_non_ascii(s)
        self.assertEqual(len(clean), 10)

    def test_unicode2str(self):
        if sys.version_info.major < 3:
            self.assertEqual(type(unicode2str("a")), str)
        else:
            self.assertEqual(type(unicode2str("a")), str)


if __name__ == "__main__":
    unittest.main()
