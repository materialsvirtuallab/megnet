import unittest
import numpy as np

from megnet.utils.general_utils import expand_1st, to_list


class TestGeneralUtils(unittest.TestCase):
    def test_expand_dim(self):
        x = np.array([1, 2, 3])
        self.assertListEqual(list(expand_1st(x).shape), [1, 3])

    def test_to_list(self):
        x = 1
        y = [1]
        self.assertListEqual(to_list(x), [1])
        self.assertListEqual(to_list(y), y)

if __name__ == "__main__":
    unittest.main()
