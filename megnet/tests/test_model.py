import unittest
import numpy as np
from megnet.model import set2set_model, set2set_with_embedding_mp
from megnet.callbacks import ModelCheckpointMAE, GeneratorLog, ManualStop
from glob import glob
import os


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_feature = 3
        cls.n_bond_features = 6
        cls.n_global_features = 2

        def generator(x, y):
            while True:
                yield x, y

        x_crystal = [np.array([1, 2, 3, 4]).reshape((1, -1)),
                     np.random.normal(size=(1, 6, cls.n_bond_features)),
                     np.random.normal(size=(1, 2, cls.n_global_features)),
                     np.array([[0, 0, 1, 1, 2, 3]]),
                     np.array([[1, 1, 0, 0, 3, 2]]),
                     np.array([[0, 0, 1, 1]]),
                     np.array([[0, 0, 0, 0, 1, 1]]),
                     ]
        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_crystal = generator(x_crystal, y)
        x_mol = [np.random.normal(size=(1, 4, cls.n_feature)),
                 np.random.normal(size=(1, 6, cls.n_bond_features)),
                 np.random.normal(size=(1, 2, cls.n_global_features)),
                 np.array([[0, 0, 1, 1, 2, 3]]),
                 np.array([[1, 1, 0, 0, 3, 2]]),
                 np.array([[0, 0, 1, 1]]),
                 np.array([[0, 0, 0, 0, 1, 1]]),
                 ]
        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_mol = generator(x_mol, y)

    def test_crystal_model(self):
        model = set2set_with_embedding_mp(self.n_bond_features, self.n_global_features, n_blocks=1, lr=1e-2,
                                          n1=4, n2=4, n3=4, n_pass=1, n_target=1)
        callbacks = [ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5',
                                        save_best_only=True,
                                        val_gen=self.train_gen_crystal,
                                        steps_per_val=1,
                                        is_pa=False),
                     GeneratorLog(self.train_gen_crystal, 1,
                                  self.train_gen_crystal, 1,
                                  val_names=['Ef'], val_units=['eV/atom']),
                     ManualStop()]

        model.fit_generator(generator=self.train_gen_crystal, steps_per_epoch=1, epochs=2, verbose=1,
                            callbacks=callbacks)
        model_files = glob('val_mae*.hdf5')
        self.assertGreater(len(model_files), 0)
        for i in model_files:
            os.remove(i)

    def test_molecule_model(self):
        model = set2set_model(self.n_feature, self.n_bond_features, self.n_global_features,
                              n_blocks=1, lr=1e-2, n1=4, n2=4, n3=4, n_pass=1, n_target=1)
        callbacks = [ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5',
                                        save_best_only=True,
                                        val_gen=self.train_gen_mol,
                                        steps_per_val=1,
                                        is_pa=False),
                     GeneratorLog(self.train_gen_mol, 1,
                                  self.train_gen_mol, 1,
                                  val_names=['Ef'], val_units=['eV/atom']),
                     ManualStop()]

        model.fit_generator(generator=self.train_gen_mol, steps_per_epoch=1, epochs=2, verbose=1, callbacks=callbacks)
        model_files = glob('val_mae*.hdf5')
        self.assertGreater(len(model_files), 0)
        for i in model_files:
            os.remove(i)


if __name__ == "__main__":
    unittest.main()
