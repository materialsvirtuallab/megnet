import unittest
from keras.models import Model
from keras.layers import Input, Dense
from megnet.callbacks import GeneratorLog, ModelCheckpointMAE, ManualStop, ReduceLRUponNan
from megnet.layers import MEGNetLayer
import numpy as np
from io import StringIO
import sys
import os
import glob
import keras.backend as K
from keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return self.x, self.y


class TestCallBack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_feature = 5
        cls.n_bond_features = 6
        cls.n_global_features = 2
        cls.x = [
            Input(shape=(None, cls.n_feature)),
            Input(shape=(None, cls.n_bond_features)),
            Input(shape=(None, cls.n_global_features)),
            Input(shape=(None,), dtype='int32'),
            Input(shape=(None,), dtype='int32'),
            Input(shape=(None,), dtype='int32'),
            Input(shape=(None,), dtype='int32'),
        ]
        units_v = [2, 2]
        units_e = [2, 2]
        units_u = [2, ]
        layer = MEGNetLayer(units_v, units_e, units_u)
        out = layer(cls.x)
        out = Dense(1)(out[2])
        cls.model = Model(inputs=cls.x, outputs=out)
        cls.model.compile(loss='mse', optimizer='adam')
        cls.x = [np.random.normal(size=(1, 4, cls.n_feature)),
                 np.random.normal(size=(1, 6, cls.n_bond_features)),
                 np.random.normal(size=(1, 2, cls.n_global_features)),
                 np.array([[0, 0, 1, 1, 2, 3]]),
                 np.array([[1, 1, 0, 0, 3, 2]]),
                 np.array([[0, 0, 1, 1]]),
                 np.array([[0, 0, 0, 0, 1, 1]]),
                 ]
        cls.y = np.random.normal(size=(1, 2, 1))
        cls.train_gen = Generator(cls.x, cls.y)

    def test_callback(self):
        callbacks = [GeneratorLog(self.train_gen, steps_per_train=1, val_gen=self.train_gen, steps_per_val=1,
                                  n_every=1, val_names=['conductivity'], val_units=['S/cm']),
                     ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5', val_gen=self.train_gen,
                                        steps_per_val=1),
                     ]
        captured_output = StringIO()
        sys.stdout = captured_output

        before_fit_file = glob.glob("./val_mae*.hdf5")
        self.model.fit_generator(self.train_gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
        after_fit_file = glob.glob("./val_mae*.hdf5")
        sys.stdout = sys.__stdout__
        result = captured_output.getvalue()
        self.assertRegex(result, "Train MAE:\nconductivity: [-+]?\d*\.\d+|\d+ S/cm")
        self.assertRegex(result, "Test MAE:\nconductivity: [-+]?\d*\.\d+|\d+ S/cm")

        self.assertEqual(len(before_fit_file), 0)
        self.assertEqual(len(after_fit_file), 1)
        os.remove(after_fit_file[0])

    def test_manual_stop(self):
        callbacks = [ManualStop()]
        epoch_count = 0
        for i in range(3):
            if not self.model.stop_training:
                self.model.fit_generator(self.train_gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
                epoch_count += 1
        self.assertEqual(epoch_count, 3)
        open('STOP', 'a').close()
        for i in range(3):
            if not self.model.stop_training:
                self.model.fit_generator(self.train_gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
                epoch_count += 1
        self.assertEqual(epoch_count, 4)
        os.remove('STOP')

    def test_reduce_lr_upon_nan(self):
        callbacks = [ReduceLRUponNan(patience=100)]
        self.assertAlmostEqual(float(K.get_value(self.model.optimizer.lr)), 1e-3)
        gen = Generator(self.x, np.array([1, np.nan]).reshape((1, 2, 1)))
        self.model.fit_generator(gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
        self.assertAlmostEqual(float(K.get_value(self.model.optimizer.lr)), 0.5e-3)


if __name__ == "__main__":
    unittest.main()
