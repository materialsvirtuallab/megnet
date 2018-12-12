import unittest
from megnet.losses import mean_squared_error_with_scale
import numpy as np
import tensorflow as tf


class TestLosses(unittest.TestCase):
    def test_mse(self):
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([0.05, 0.15, 0.25])
        loss = mean_squared_error_with_scale(x, y, scale=100)
        sess = tf.InteractiveSession()
        self.assertAlmostEqual(sess.run(loss), np.mean((x-y)**2) * 100)


if __name__ == '__main__':
    unittest.main()
