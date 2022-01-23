import unittest

from monty.fractions import gcd, lcm, gcd_float


class FuncTestCase(unittest.TestCase):
    def test_gcd(self):
        self.assertEqual(gcd(7, 14, 63), 7)

    def test_lcm(self):
        self.assertEqual(lcm(2, 3, 4), 12)

    def test_gcd_float(self):
        vs = [6.2, 12.4, 15.5 + 5e-9]
        self.assertAlmostEqual(gcd_float(vs, 1e-8), 3.1)


if __name__ == "__main__":
    unittest.main()
