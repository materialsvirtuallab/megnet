from megnet.layers import MEGNetLayer
from keras.layers import Input
from keras.models import Model
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_features = 5
        cls.n_bond_features = 6
        cls.n_global_features = 2
        cls.x = [
            Input(shape=(None, cls.n_features)),
            Input(shape=(None, cls.n_bond_features)),
            Input(shape=(None, cls.n_global_features)),
            Input(shape=(None, ), dtype='int32'),
            Input(shape=(None, ), dtype='int32'),
            Input(shape=(None, ), dtype='int32'),
            Input(shape=(None, ), dtype='int32'),
        ]

    def test_megnet(self):
        units_v = [13, 14, 16]
        units_e = [16, 16, 17]
        units_u = [13, 14, 18]
        layer = MEGNetLayer(units_v, units_e, units_u)
        out = layer(self.x)
        self.assertListEqual([i._keras_shape[-1] for i in out], [units_v[-1], units_e[-1], units_u[-1]])
        new_layer = MEGNetLayer.from_config(layer.get_config())
        out2 = new_layer(self.x)
        self.assertListEqual([i._keras_shape[-1] for i in out2], [units_v[-1], units_e[-1], units_u[-1]])

        int32 = 'int32'
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
        out = MEGNetLayer([10, 5], [20, 4], [30, 3])(
            [x1_, x2_, x3_, x4_, x5_, x6_, x7_])
        model = Model(inputs=[x1_, x2_, x3_, x4_, x5_, x6_, x7_], outputs=out)
        model.compile('adam', 'mse')
        ans = model.predict([x1, x2, x3, x4, x5, x6, x7])
        self.assertEqual(ans[0].shape, (1, 5, 5))


if __name__ == "__main__":
    unittest.main()
