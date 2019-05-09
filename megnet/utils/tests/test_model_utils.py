from megnet.utils.model_utils import QM9Model, pb
import unittest


class TestModelUtil(unittest.TestCase):
    @unittest.skipIf(pb is None, "Molecule module requires openbabel")
    def test_homo_qm9(self):
        model = QM9Model('HOMO')
        self.assertAlmostEqual(model.predict_smiles('C'), -10.557696, 3)


if __name__ == "__main__":
    unittest.main()
