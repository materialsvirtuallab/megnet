import unittest
from megnet.utils.molecule import MEGNetMolecule
from megnet.models import MEGNetModel
import os

CWD = os.path.dirname(os.path.abspath(__file__))


class TestMolecule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.molecule = MEGNetMolecule(['C', 'O', 'O'], [[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        cls.model = MEGNetModel.from_file(os.path.join(CWD, '../../../mvl_models/mp-2019.4.1/formation_energy.hdf5'))

    def test_mol(self):
        pred = self.model.predict_structure(self.molecule)
        self.assertAlmostEqual(pred, -0.10959488)


if __name__ == "__main__":
    unittest.main()