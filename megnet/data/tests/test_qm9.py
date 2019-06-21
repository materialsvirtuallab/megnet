import unittest
from unittest.mock import MagicMock
import json
from megnet.data.qm9 import load_qm9_faber, ring_to_vector, FeatureClean, Scaler, sublist_from_qm9, AtomNumberToTypeConverter
import os


module_dir = os.path.dirname(os.path.abspath(__file__))


class QM9Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(module_dir, 'qm9','qm9.json'), 'r') as f:
            cls.data = json.load(f)
        cls.db_connection = type("MockTest", (), {})
        cls.db_connection.find = MagicMock(return_value=cls.data)

    def test_atom_number_converter(self):
        anc = AtomNumberToTypeConverter()
        types = {1: 1, 6: 2, 7: 4, 8: 6, 9: 8}
        keys = list(types.keys())
        values = list(types.values())
        self.assertListEqual(anc.convert(keys), values)

    def test_load_qm9(self):
        features_list, connection_list, global_list, index1_list, index2_list, targets = load_qm9_faber(
            self.db_connection)
        data = [features_list, connection_list, global_list, index1_list, index2_list, targets]
        # 11 molecules
        self.assertListEqual([len(i) for i in data[:-1]], [11] * 5)

        # first molecule has 5 atoms and 20 bonds
        self.assertEqual(len(features_list[0]), 5)
        self.assertEqual(len(connection_list[0]), 20)
        self.assertEqual(len(global_list[0]), 1)
        self.assertEqual(len(index1_list[0]), 20)
        self.assertEqual(len(index2_list[0]), 20)
        # target dimension is 11*19
        self.assertListEqual(list(targets.shape), [11, 19])

    def test_ring_to_vector(self):
        x = [2, 2, 3]
        expected = [0, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected, ring_to_vector(x))

    def test_feature_clean(self):
        features_list, connection_list, global_list, index1_list, index2_list, targets = load_qm9_faber(
            self.db_connection)
        fc = FeatureClean()
        out = fc.fit_transform(features_list)
        # first molecule has 5 atoms and final feature number is 21
        self.assertListEqual(list(out[0].shape), [5, 21])

        fc = FeatureClean(categorical=["bond_type", "same_ring"],
                          feature_labels=["bond_type", 'graph_distance', 'same_ring', "spatial_distance"])
        out = fc.fit_transform(connection_list)

        # note that the bond feature dimension is 27 for the whole dataset
        # here we are missing 1 element (totally 5), thus the dimension is
        # 4 + 1 + 1 + 100 = 106
        self.assertListEqual(list(out[0].shape), [20, 106])

    def test_scaler(self):
        x = [[[1, 2], [1, 2]], [[3, 4], [5, 6]]]
        scaler = Scaler()
        out = scaler.fit_transform(x)
        self.assertTrue(isinstance(out, list))
        self.assertListEqual(list(out[0].shape), [2, 2])

    def test_sublist_from_qm9(self):
        features_list, connection_list, global_list, \
            index1_list, index2_list, targets = load_qm9_faber(
                self.db_connection)
        ids = ['000002', '000001']
        t, new_features = sublist_from_qm9(ids, targets, features_list, connection_list)
        self.assertEqual(t.shape[0], 2)
        self.assertListEqual(list(t.index), ids)


if __name__ == "__main__":
    unittest.main()
