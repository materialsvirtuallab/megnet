import unittest

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from megnet.layers import Set2Set


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = Input(shape=(None, 6))
        cls.index = Input(shape=(None,), dtype="int32")

    def test_set2set(self):
        n_hidden = 10
        set2set = Set2Set(n_hidden=n_hidden)
        out = set2set([self.x, self.index])
        shapes = set2set.compute_output_shape([self.x.shape, self.index.shape])
        self.assertEqual(shapes[-1], n_hidden * 2)

        model = Model(inputs=[self.x, self.index], outputs=out)
        model.compile(loss="mse", optimizer="adam")
        x = np.random.normal(size=(1, 5, 6))
        index = np.array([[0, 0, 0, 1, 1]])
        result = model.predict([x, index])
        self.assertListEqual(list(result.shape), [1, 2, n_hidden * 2])


if __name__ == "__main__":
    unittest.main()
