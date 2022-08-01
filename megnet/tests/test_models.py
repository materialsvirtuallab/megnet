import os
import shutil
import unittest
from glob import glob

import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Lattice, Structure
from pymatgen.util.testing import PymatgenTest
from tensorflow.keras.utils import Sequence

from megnet.callbacks import ManualStop, ModelCheckpointMAE
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import GraphModel, MEGNetModel

cwd = os.path.dirname(os.path.abspath(__file__))


class TestModel(PymatgenTest):
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
                return self.x, self.y

        x_crystal = [
            np.array([1, 2, 3, 4]).reshape((1, -1)),
            np.random.normal(size=(1, 6, cls.n_bond_features)),
            np.random.normal(size=(1, 2, cls.n_global_features)),
            np.array([[0, 0, 1, 1, 2, 3]]),
            np.array([[1, 1, 0, 0, 3, 2]]),
            np.array([[0, 0, 1, 1]]),
            np.array([[0, 0, 0, 0, 1, 1]]),
        ]

        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_crystal = Generator(x_crystal, y)
        x_mol = [
            np.random.normal(size=(1, 4, cls.n_feature)),
            np.random.normal(size=(1, 6, cls.n_bond_features)),
            np.random.normal(size=(1, 2, cls.n_global_features)),
            np.array([[0, 0, 1, 1, 2, 3]]),
            np.array([[1, 1, 0, 0, 3, 2]]),
            np.array([[0, 0, 1, 1]]),
            np.array([[0, 0, 0, 0, 1, 1]]),
        ]
        y = np.random.normal(size=(1, 2, 1))
        cls.train_gen_mol = Generator(x_mol, y)

        cls.model = MEGNetModel(
            10,
            2,
            nblocks=1,
            lr=1e-2,
            n1=4,
            n2=4,
            n3=4,
            npass=1,
            ntarget=1,
            graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)),
        )
        cls.model2 = MEGNetModel(
            10,
            2,
            nblocks=1,
            lr=1e-2,
            n1=4,
            n2=4,
            n3=4,
            npass=1,
            ntarget=2,
            graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)),
        )

    def test_train_pred_w_sample_weights(self):
        s = Structure.from_file(os.path.join(cwd, "../data/tests/cifs/BaTiO3_mp-2998_computed.cif"))
        structures = [s.copy(), s.copy(), s.copy(), s.copy()]
        targets = [0.1, 0.1, 0.1, 0.1]
        with ScratchDir("."):
            self.model.train(
                structures,
                targets,
                validation_structures=structures[:2],
                validation_targets=[0.1, 0.1],
                sample_weights=[0.1, 0.2, 0.3, 0.4],
                batch_size=2,
                epochs=1,
                verbose=2,
            )
            preds = self.model.predict_structure(structures[0])
        self.assertTrue(preds.shape == (1,))

    def test_predicts(self):
        s = Structure.from_file(os.path.join(cwd, "../data/tests/cifs/BaTiO3_mp-2998_computed.cif"))
        s2 = Structure(Lattice.cubic(3.1), ["Mo", "Mo"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        predicted = self.model.predict_structures([s, s2, s, s2, s, s2])
        # make sure order is correct
        self.assertArrayAlmostEqual(predicted[0], predicted[2])
        self.assertArrayAlmostEqual(predicted[0], predicted[4])

    def test_train_pred(self):
        s = Structure.from_file(os.path.join(cwd, "../data/tests/cifs/BaTiO3_mp-2998_computed.cif"))
        structures = [s.copy(), s.copy(), s.copy(), s.copy()]
        targets = [0.1, 0.1, 0.1, 0.1]
        with ScratchDir("."):
            self.model.train(
                structures,
                targets,
                validation_structures=structures[:2],
                validation_targets=[0.1, 0.1],
                batch_size=2,
                epochs=1,
                verbose=2,
            )
            preds = self.model.predict_structure(structures[0])

            # isolated atom error
            for s in structures[3:]:
                s.apply_strain(3)
            with self.assertRaises(RuntimeError) as context:
                self.model.train(structures, targets, epochs=1, verbose=2, scrub_failed_structures=False)
                self.assertTrue("Isolated atoms found" in str(context.exception))

            with self.assertRaises(Exception) as context:
                self.model.train(structures, targets, epochs=1, verbose=2, scrub_failed_structures=True)
                self.assertTrue("structure with index" in str(context.exception))

            if os.path.isdir("callback"):
                shutil.rmtree("callback")
            self.assertTrue(np.size(preds) == 1)

    def test_single_atom_structure(self):
        s = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
        with ScratchDir("."):
            # initialize the model
            self.model.train([s, s], [0.1, 0.1], epochs=1)
            pred = self.model.predict_structure(s)
            self.assertEqual(len(pred.ravel()), 1)

    def test_two_targets(self):
        s = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
        with ScratchDir("."):
            # initialize the model
            self.model2.train([s, s], [[0.1, 0.2], [0.1, 0.2]], epochs=1)
            pred = self.model2.predict_structure(s)
            self.assertEqual(len(pred.ravel()), 2)

    def test_save_and_load(self):
        weights1 = self.model.get_weights()
        with ScratchDir("."):
            self.model.metadata = {"units": "eV"}  # This is just a random
            self.model.save_model("test.hdf5")
            model2 = GraphModel.from_file("test.hdf5")
            self.assertEqual(model2.metadata, {"units": "eV"})
        weights2 = model2.get_weights()
        self.assertTrue(np.allclose(weights1[0], weights2[0]))

    def test_check_dimension(self):
        gc = CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 20), 0.5))
        s = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
        graph = gc.convert(s)
        model = MEGNetModel(
            10,
            2,
            nblocks=1,
            lr=1e-2,
            n1=4,
            n2=4,
            n3=4,
            npass=1,
            ntarget=1,
            graph_converter=gc,
        )
        with self.assertRaises(Exception) as context:
            model.check_dimension(graph)
            self.assertTrue("The data dimension for bond" in str(context.exception))

    def test_crystal_model(self):
        callbacks = [
            ModelCheckpointMAE(
                filepath="./val_mae_{epoch:05d}_{val_mae:.6f}.hdf5",
                save_best_only=True,
                val_gen=self.train_gen_crystal,
                steps_per_val=1,
            ),
            ManualStop(),
        ]
        with ScratchDir("."):
            self.model.fit(self.train_gen_crystal, steps_per_epoch=1, epochs=2, verbose=1, callbacks=callbacks)
            model_files = glob("val_mae*.hdf5")
            self.assertGreater(len(model_files), 0)
            for i in model_files:
                os.remove(i)

    def test_crystal_model_v2(self):
        cg = CrystalGraph()
        s = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
        with ScratchDir("."):
            model = MEGNetModel(
                nfeat_edge=None,
                nfeat_global=2,
                nblocks=1,
                lr=1e-2,
                n1=4,
                n2=4,
                n3=4,
                npass=1,
                ntarget=1,
                graph_converter=cg,
                centers=np.linspace(0, 4, 10),
                width=0.5,
            )
            model = model.train([s, s], [0.1, 0.1], epochs=2)
            t = model.predict_structure(s)
            self.assertTrue(t.shape == (1,))

    def test_from_url(self):
        with ScratchDir("."):
            model = MEGNetModel.from_url(
                "https://github.com/materialsvirtuallab/megnet/raw/master/mvl_models/mp-2019.4.1/formation_energy.hdf5"
            )
            li2o = self.get_structure("Li2O")
            self.assertAlmostEqual(float(model.predict_structure(li2o)), -2.0152957439422607, places=4)

    def test_from_mvl_models(self):
        with ScratchDir("."):
            model = MEGNetModel.from_mvl_models("Eform_MP_2019")
            li2o = self.get_structure("Li2O")
            self.assertAlmostEqual(float(model.predict_structure(li2o)), -2.0152957439422607, places=4)


if __name__ == "__main__":
    unittest.main()
