"""
Test the data types
"""
import unittest

import numpy as np
import tensorflow as tf

from megnet.config import DataType, set_global_dtypes


class TestDataType(unittest.TestCase):
    def test_set_16(self):
        DataType.set_dtype("16")
        self.assertTrue(DataType.np_int, np.int16)
        self.assertTrue(DataType.np_float, np.float16)
        self.assertTrue(DataType.tf_int, tf.int16)
        self.assertTrue(DataType.tf_float, tf.float16)
        set_global_dtypes("32")
        self.assertTrue(DataType.np_int, np.int32)

    def test_set_32(self):
        DataType.set_dtype("32")
        self.assertTrue(DataType.np_int, np.int32)
        self.assertTrue(DataType.np_float, np.float32)

    def test_wrong(self):
        self.assertRaises(ValueError, DataType.set_dtype, "64")


if __name__ == "__main__":
    unittest.main()
