import tensorflow as tf
from megnet.layers import LinearWithIndex
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import unittest


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = Input(shape=(None, 6))
        cls.index = Input(shape=(None,), dtype='int32')

    def test_linear(self):
        linear = LinearWithIndex(mode='mean')
        out = linear([self.x, self.index])
        shapes = linear.compute_output_shape([self.x.shape, self.index.shape])
        self.assertEqual(shapes[-1], 6)
        model = Model(inputs=[self.x, self.index], outputs=out)
        model.compile(loss='mse', optimizer='adam')
        x = np.random.normal(size=(1, 5, 6))
        expected_output = np.concatenate([np.mean(x[0, :3, :], axis=0, keepdims=True),
                                          np.mean(x[0, 3:, :], axis=0, keepdims=True)], axis=0)
        index = np.array([[0, 0, 0, 1, 1]])
        result = model.predict([x, index])
        diff = np.linalg.norm(result[0, :, :] - expected_output)
        self.assertListEqual(list(result.shape), [1, 2, 6])
        self.assertTrue(diff < 1e-5)

        linear = LinearWithIndex(mode='sum')
        out = linear([self.x, self.index])
        model = Model(inputs=[self.x, self.index], outputs=out)
        model.compile(loss='mse', optimizer='adam')
        x = np.random.normal(size=(1, 5, 6))
        expected_output = np.concatenate([np.sum(x[0, :3, :], axis=0, keepdims=True),
                                          np.sum(x[0, 3:, :], axis=0, keepdims=True)], axis=0)
        index = np.array([[0, 0, 0, 1, 1]])
        result = model.predict([x, index])
        diff = np.linalg.norm(result[0, :, :] - expected_output)
        self.assertListEqual(list(result.shape), [1, 2, 6])
        self.assertTrue(diff < 1e-5)


if __name__ == "__main__":
    unittest.main()
