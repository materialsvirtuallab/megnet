# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
#TODO: Replace with proper module doc.
"""

import unittest

from monty.itertools import iterator_from_slice


class FuncTest(unittest.TestCase):
    def test_iterator_from_slice(self):
        self.assertEqual(list(iterator_from_slice(slice(0, 6, 2))), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
