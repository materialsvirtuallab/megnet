import tensorflow as tf
import unittest
from megnet.activations import softplus2
import numpy as np


def softplus_np(x):
    return np.log(np.exp(x) + 1) - np.log(2.)


class TestSP(unittest.TestCase):
    def test_softplus(self):
        x = 10.0
        self.assertAlmostEqual(softplus2(x).numpy(), softplus_np(x), places=5)


if __name__ == '__main__':
    unittest.main()
