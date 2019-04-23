import unittest
import numpy as np
from megnet.models import MEGNetModel, GraphModel
from megnet.callbacks import ModelCheckpointMAE, GeneratorLog, ManualStop
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from glob import glob
import os
from pymatgen import Structure, Lattice
import shutil
from monty.tempfile import ScratchDir
from keras.utils import Sequence

cwd = os.path.dirname(os.path.abspath(__file__))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_feature = 3
        cls.n_bond_features = 10
        cls.n_global_features = 2

        class Generator(Sequence):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __len__(self):
                return 10
            def __getitem__(self, index):
                return  self.x, self.y

        x_crystal = [np.array([1, 2, 3, 4]).reshape((1, -1)),
                     np.random.normal(size=(1, 6, cls.n_bond_features)),
                     np.random.normal(size=(1, 2, cls.n_global_features)),
                     np.array([[0, 0, 1, 1, 2, 3]]),
                     np.array([[1, 1, 0, 0, 3, 2]]),
                     np.array([[0, 0, 1, 1]]),
                     np.array([[0, 0, 0, 0, 1, 1]]),
                     ]

        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_crystal = Generator(x_crystal, y)
        x_mol = [np.random.normal(size=(1, 4, cls.n_feature)),
                 np.random.normal(size=(1, 6, cls.n_bond_features)),
                 np.random.normal(size=(1, 2, cls.n_global_features)),
                 np.array([[0, 0, 1, 1, 2, 3]]),
                 np.array([[1, 1, 0, 0, 3, 2]]),
                 np.array([[0, 0, 1, 1]]),
                 np.array([[0, 0, 0, 0, 1, 1]]),
                 ]
        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_mol = Generator(x_mol, y)

        cls.model = MEGNetModel(10, 2, nblocks=1, lr=1e-2,
                                n1=4, n2=4, n3=4, npass=1, ntarget=1,
                                graph_convertor=CrystalGraph(bond_convertor=GaussianDistance(np.linspace(0, 5, 10), 0.5)),
                                )

    def test_train_pred(self):
        s = Structure.from_file(os.path.join(cwd, '../data/tests/cifs/BaTiO3_mp-2998_computed.cif'))
        structures = [s] * 4
        targets = [0.1, 0.1, 0.1, 0.1]
        self.model.train(structures,
                         targets,
                         validation_structures=structures[:2],
                         validation_targets=[0.1, 0.1],
                         batch_size=2,
                         epochs=1,
                         verbose=2)
        preds = self.model.predict_structure(structures[0])
        if os.path.isdir('callback'):
            shutil.rmtree('callback')
        self.assertTrue(np.size(preds) == 1)

    def test_single_atom_structure(self):
        s = Structure(Lattice.cubic(3), ['Si'], [[0, 0, 0]])
        # initialize the model
        self.model.train([s, s], [0.1, 0.1], epochs=1)
        pred = self.model.predict_structure(s)
        self.assertEqual(len(pred.ravel()), 1)

    def test_save_and_load(self):
        weights1 = self.model.get_weights()
        with ScratchDir('.'):
            self.model.save_model('test.hdf5')
            model2 = GraphModel.from_file('test.hdf5')
        weights2 = model2.get_weights()
        self.assertTrue(np.allclose(weights1[0], weights2[0]))

    def test_crystal_model(self):
        callbacks = [ModelCheckpointMAE(filepath='./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5',
                                        save_best_only=True,
                                        val_gen=self.train_gen_crystal,
                                        steps_per_val=1,
                                        is_pa=False),
                     GeneratorLog(self.train_gen_crystal, 1,
                                  self.train_gen_crystal, 1,
                                  val_names=['Ef'], val_units=['eV/atom']),
                     ManualStop()]

        self.model.fit_generator(generator=self.train_gen_crystal, steps_per_epoch=1, epochs=2, verbose=1,
                                 callbacks=callbacks)
        model_files = glob('val_mae*.hdf5')
        self.assertGreater(len(model_files), 0)
        for i in model_files:
            os.remove(i)


if __name__ == "__main__":
    unittest.main()
