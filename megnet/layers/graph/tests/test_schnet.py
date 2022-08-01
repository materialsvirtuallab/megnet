import unittest

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from megnet.layers import InteractionLayer


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_features = 10
        cls.n_bond_features = 6
        cls.n_global_features = 2
        cls.x = [
            Input(shape=(None, cls.n_features)),
            Input(shape=(None, cls.n_bond_features)),
            Input(shape=(None, cls.n_global_features)),
            Input(shape=(None,), dtype="int32"),
            Input(shape=(None,), dtype="int32"),
            Input(shape=(None,), dtype="int32"),
            Input(shape=(None,), dtype="int32"),
        ]

    def test_layer(self):
        layer = InteractionLayer()
        out = layer(self.x)
        self.assertListEqual(
            [i.shape[-1] for i in out], [self.n_features, self.n_bond_features, self.n_global_features]
        )
        new_layer = InteractionLayer.from_config(layer.get_config())
        out2 = new_layer(self.x)
        self.assertListEqual(
            [i.shape[-1] for i in out2], [self.n_features, self.n_bond_features, self.n_global_features]
        )

        int32 = "int32"
        x1 = np.random.rand(1, 5, 10)
        x2 = np.random.rand(1, 6, 5)
        x3 = np.random.rand(1, 2, 20)
        x4 = np.array([0, 1, 2, 3, 3, 4]).reshape([1, -1])
        x5 = np.array([1, 0, 3, 2, 4, 3]).reshape([1, -1])
        x6 = np.array([[0, 0, 1, 1, 1]])
        x7 = np.array([[0, 0, 1, 1, 1, 1]])
        x1_ = Input(shape=(None, 10))
        x2_ = Input(shape=(None, 5))
        x3_ = Input(shape=(None, 20))
        x4_ = Input(shape=(None,), dtype=int32)
        x5_ = Input(shape=(None,), dtype=int32)
        x6_ = Input(shape=(None,), dtype=int32)
        x7_ = Input(shape=(None,), dtype=int32)
        out = InteractionLayer()([x1_, x2_, x3_, x4_, x5_, x6_, x7_])
        model = Model(inputs=[x1_, x2_, x3_, x4_, x5_, x6_, x7_], outputs=out)
        model.compile("adam", "mse")
        ans = model.predict([x1, x2, x3, x4, x5, x6, x7])
        self.assertEqual(ans[0].shape, (1, 5, 10))


if __name__ == "__main__":
    unittest.main()
