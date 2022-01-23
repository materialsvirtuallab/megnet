import unittest

from monty.bisect import index, find_lt, find_le, find_gt, find_ge


class FuncTestCase(unittest.TestCase):
    def test_funcs(self):
        l = [0, 1, 2, 3, 4]
        self.assertEqual(index(l, 1), 1)
        self.assertEqual(find_lt(l, 1), 0)
        self.assertEqual(find_gt(l, 1), 2)
        self.assertEqual(find_le(l, 1), 1)
        self.assertEqual(find_ge(l, 2), 2)
        # self.assertEqual(index([0, 1, 1.5, 2], 1.501, atol=0.1), 4)


if __name__ == "__main__":
    unittest.main()
