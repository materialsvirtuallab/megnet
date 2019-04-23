from megnet.utils.model_utils import QM9Model
import unittest


class TestModelUtil(unittest.TestCase):

    def test_homo_qm9(self):
        model = QM9Model('HOMO')
        self.assertAlmostEqual(model.predict_smiles('C'), -10.557696)


if __name__ == "__main__":
    unittest.main()
