import unittest
from megnet.data.graph import GaussianDistance, ClassGenerator, expand_1st
import numpy as np


class TestGraph(unittest.TestCase):
    def test_gaussian_expansion(self):
        x = np.random.normal(size=(10, ))
        gd = GaussianDistance()
        out = gd.convert(x)
        self.assertListEqual(list(out.shape), [10, 20])

    def test_class_generator(self):
        feature = [np.random.normal(size=(3, 4)), np.random.normal(size=(2, 4))]
        bond = [np.random.normal(size=(2, 5)), np.random.normal(size=(1, 5))]
        glob_features = [np.random.normal(size=(1, 2 )), np.random.normal(size=(1, 2))]
        index1 = [np.array([0, 1]), np.array([0])]
        index2 = [np.array([1, 2]), np.array([1])]
        targets = np.random.normal(size=(2, 1))
        gen = ClassGenerator(feature, bond, glob_features, index1, index2, targets, batch_size=2)
        data = next(gen)
        self.assertListEqual(list(data[0][0].shape), [1, 5, 4])
        self.assertListEqual(list(data[0][1].shape), [1, 3, 5])
        self.assertListEqual(list(data[0][2].shape), [1, 2, 2])
        self.assertListEqual(list(data[0][3].shape), [1, 3])
        self.assertListEqual(list(data[0][4].shape), [1, 3])
        self.assertListEqual(list(data[1].shape), [1, 2, 1])

    def test_expand_dim(self):
        x = np.array([1, 2, 3])
        self.assertListEqual(list(expand_1st(x).shape), [1, 3])

if __name__ == "__main__":
    unittest.main()
