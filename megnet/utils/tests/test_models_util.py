from megnet.utils.models import load_model
from megnet.models import GraphModel
import unittest


class TestLoadModel(unittest.TestCase):

    def test_load_crystal(self):
        model = load_model('Eform_MP_2019')
        self.assertIsInstance(model, GraphModel)
        with self.assertRaises(ValueError):
            _ = load_model('Eform_MP_2020')

    def test_load_qm9(self):
        model = load_model('QM9_G_2018')
        self.assertIsInstance(model, GraphModel)


if __name__ == "__main__":
    unittest.main()
