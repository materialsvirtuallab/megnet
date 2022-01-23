import unittest

from monty.inspect import *


class LittleCatA:
    pass


class LittleCatB(LittleCatA):
    pass


class LittleCatC:
    pass


class LittleCatD(LittleCatB):
    pass


class InspectTest(unittest.TestCase):
    def test_func(self):
        # Not a real test. Need something better.
        self.assertTrue(find_top_pyfile())
        self.assertTrue(caller_name())

    def test_all_subclasses(self):
        self.assertEqual(all_subclasses(LittleCatA), [LittleCatB, LittleCatD])


if __name__ == "__main__":
    unittest.main()
