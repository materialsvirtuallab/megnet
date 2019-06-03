import unittest
import numpy as np

from megnet.utils.general import expand_1st, to_list


class TestGeneralUtils(unittest.TestCase):
    def test_expand_dim(self):
        x = np.array([1, 2, 3])
        self.assertListEqual(list(expand_1st(x).shape), [1, 3])

    def test_to_list(self):
        x = 1
        y = [1]
        z = tuple([1, 2, 3])
        v = np.array([1, 2, 3])
        for k in [x, y, z, v]:
            self.assertTrue(type(to_list(k)), list)


if __name__ == "__main__":
    unittest.main()
