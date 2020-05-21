import unittest
import numpy as np

from megnet.utils.general import (
    expand_1st, to_list, check_shape, reshape, fast_label_binarize)


class TestGeneralUtils(unittest.TestCase):
    def test_expand_dim(self):
        x = np.array([1, 2, 3])
        self.assertListEqual(list(expand_1st(x).shape), [1, 3])

    def test_to_list(self):
        x = 1
        y = [1]
        z = tuple([1, 2, 3])
        v = np.array([1, 2, 3])
        k = np.array([[1, 2], [3, 4]])
        for k in [x, y, z, v, k]:
            self.assertTrue(type(to_list(k)), list)

    def test_fast_label_binarize(self):
        binaries = fast_label_binarize(1, [0, 1])
        self.assertListEqual(binaries, [0])
        binaries = fast_label_binarize(1, [0, 1, 2])
        self.assertListEqual(binaries, [0, 1, 0])

    def test_check_shape(self):
        x = np.random.normal(size=(10, 20))
        self.assertTrue(check_shape(x, [10, 20, None]))
        self.assertTrue(check_shape(x, [10, 20]))
        self.assertFalse(check_shape(x, [10, 10]))
        self.assertTrue(check_shape(None, [10, 20]))

    def test_reshape(self):
        x = np.random.normal(size=(10, 20))
        self.assertEqual(reshape(x, [10, 20, None]).shape, (10, 20, 1))
        self.assertEqual(reshape(x, [10, 20, 20, None]).shape, (10, 20, 20, 1))


if __name__ == "__main__":
    unittest.main()
