import unittest

import numpy as np

from megnet.activations import softplus2, swish


def softplus_np(x):
    return np.log(np.exp(x) + 1) - np.log(2.0)


def swish_np(x):
    return x / (1 + np.exp(-x))


class TestSP(unittest.TestCase):
    def test_softplus(self):
        x = 10.0
        self.assertAlmostEqual(softplus2(x).numpy(), softplus_np(x), places=5)

    def test_swish(self):
        self.assertAlmostEqual(swish(10.0).numpy(), swish_np(10.0), places=5)


if __name__ == "__main__":
    unittest.main()
