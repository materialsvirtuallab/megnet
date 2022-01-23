import unittest
from monty.operator import operator_from_str


class OperatorTestCase(unittest.TestCase):
    def test_something(self):
        assert operator_from_str("==")(1, 1) and operator_from_str("+")(1, 1) == 2


if __name__ == "__main__":
    unittest.main()
