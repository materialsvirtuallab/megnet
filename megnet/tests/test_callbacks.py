import unittest
from keras.models import Model
from keras.layers import Input, Dense
from megnet.callbacks import GeneratorLog, ModelCheckpointMAE, ManualStop, ReduceLRUponNan
from megnet.layers import MEGNetLayer
from megnet.utils.preprocessing import StandardScaler
import numpy as np
import os
import glob
import keras.backend as kb
from keras.utils import Sequence
from monty.tempfile import ScratchDir


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
        cls.inp = [
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
        out = layer(cls.inp)
        cls.out = Dense(1)(out[2])
        cls.model = Model(inputs=cls.inp, outputs=cls.out)
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
        before_fit_file = glob.glob("./val_mae*.hdf5")
        self.model.fit_generator(self.train_gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
        after_fit_file = glob.glob("./val_mae*.hdf5")

        self.assertEqual(len(before_fit_file), 0)
        self.assertEqual(len(after_fit_file), 1)
        os.remove(after_fit_file[0])

        callback_mae = ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5', val_gen=self.train_gen,
                                          steps_per_val=1, target_scaler=StandardScaler(1, 1, is_intensive=True))

        dummy_target = np.array([[1, 1], [2, 2]])
        dummy_nb_atoms = np.array([[2], [3]])
        transformed = callback_mae.target_scaler.inverse_transform(dummy_target, dummy_nb_atoms)
        self.assertTrue(np.allclose(transformed, np.array([[2, 2], [3, 3]])))

        callback_mae = ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5', val_gen=self.train_gen,
                                          steps_per_val=1, target_scaler=StandardScaler(1, 1, is_intensive=False))

        dummy_target = np.array([[1, 1], [2, 2]])
        dummy_nb_atoms = np.array([[2], [3]])
        transformed = callback_mae.target_scaler.inverse_transform(dummy_target, dummy_nb_atoms)
        self.assertTrue(np.allclose(transformed, np.array([[4, 4], [9, 9]])))

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
        with ScratchDir('.'):
            callbacks = [ReduceLRUponNan(patience=100)]
            self.assertAlmostEqual(float(kb.get_value(self.model.optimizer.lr)), 1e-3)
            gen = Generator(self.x, np.array([1, np.nan]).reshape((1, 2, 1)))
            self.model.fit_generator(gen, steps_per_epoch=1, epochs=1, callbacks=callbacks, verbose=0)
            self.assertAlmostEqual(float(kb.get_value(self.model.optimizer.lr)), 0.5e-3)

            inp = [
                Input(shape=(None, self.n_feature)),
                Input(shape=(None, self.n_bond_features)),
                Input(shape=(None, self.n_global_features)),
                Input(shape=(None,), dtype='int32'),
                Input(shape=(None,), dtype='int32'),
                Input(shape=(None,), dtype='int32'),
                Input(shape=(None,), dtype='int32'),
            ]
            units_v = [2, 2]
            units_e = [2, 2]
            units_u = [2, ]
            layer = MEGNetLayer(units_v, units_e, units_u)
            out = layer(inp)
            out = Dense(1)(out[2])
            model = Model(inputs=inp, outputs=out)
            model.compile(loss='mse', optimizer='adam')
            x = [np.random.normal(size=(1, 4, self.n_feature)),
                     np.random.normal(size=(1, 6, self.n_bond_features)),
                     np.random.normal(size=(1, 2, self.n_global_features)),
                     np.array([[0, 0, 1, 1, 2, 3]]),
                     np.array([[1, 1, 0, 0, 3, 2]]),
                     np.array([[0, 0, 1, 1]]),
                     np.array([[0, 0, 0, 0, 1, 1]]),
                     ]
            y = np.random.normal(size=(1, 2, 1))
            train_gen = Generator(x, y)

            callbacks = [ReduceLRUponNan(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5', patience=100),
                         ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5', val_gen=train_gen,
                                            steps_per_val=1)
                         ]
            # 1. involve training and saving
            model.fit_generator(train_gen, steps_per_epoch=1, epochs=2, callbacks=callbacks, verbose=1)
            # 2. throw nan loss, trigger ReduceLRUponNan
            model.fit_generator(gen, steps_per_epoch=1, epochs=2, callbacks=callbacks, verbose=1)
            # 3. Normal training, recover saved model from 1
            model.fit_generator(train_gen, steps_per_epoch=1, epochs=2, callbacks=callbacks, verbose=1)
            self.assertAlmostEqual(float(kb.get_value(model.optimizer.lr)), 0.25e-3)


if __name__ == "__main__":
    unittest.main()
