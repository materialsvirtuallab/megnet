import unittest

import numpy as np

from megnet.layers.featurizer import GaussianExpansion


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):
        x = np.random.normal(size=(1, 10))
        centers = np.linspace(0, 6, 100)
        width = 0.5

        ge = GaussianExpansion(centers=centers, width=width)
        self.assertTrue(ge(x).shape == (1, 10, 100))

        np.testing.assert_array_equal(centers, ge.get_config()["centers"])

        self.assertTrue(ge.built)
        self.assertTrue(ge.compute_output_shape((1, 10)) == (1, 10, 100))


if __name__ == "__main__":
    unittest.main()
