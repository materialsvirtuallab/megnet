from megnet.layers import MEGNetLayer
from keras.layers import Input
import unittest


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_feature = 5
        cls.n_bond_features = 6
        cls.n_global_features = 2
        cls.x = [
            Input(shape=(None, cls.n_feature)),
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


if __name__ == "__main__":
    unittest.main()
