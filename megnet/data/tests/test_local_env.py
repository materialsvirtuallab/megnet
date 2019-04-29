import unittest
from megnet.data.local_env import MinimumDistanceNNAll, AllAtomPairs
from pymatgen import Structure, Molecule
import os
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def _equal(x, y):
    if isinstance(x, list):
        return all([_equal(i, j) for i, j in zip(x, y)])
    elif isinstance(x, dict):
        return all(_equal(x[i], y[i]) for i in x.keys())
    else:
        if x == y:
            return True
        else:
            print(x, y)
            return False


def _sort_neighbors(neighbors):
    out = []
    for n in neighbors:
        out.append([sorted(n, key=lambda x: (x['weight'], x['site_index']))])
    return out


class TestLocalEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.structure = Structure.from_file(os.path.join(MODULE_DIR, 'cifs', 'BaTiO3_mp-2998_computed.cif'))
        cls.molecule = Molecule(['C', 'O', 'O'], [[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        cls.mall = MinimumDistanceNNAll(4)
        cls.aapair = AllAtomPairs()

    def test_minimal_distance(self):
        neighbors1 = self.mall.get_all_nn_info(self.structure)
        neighbors2 = [self.mall.get_nn_info(self.structure, i) for i in range(len(self.structure))]
        self.assertTrue(_equal(_sort_neighbors(neighbors1), _sort_neighbors(neighbors2)))

    def test_all_atom_pairs(self):
        mol_pairs = self.aapair.get_all_nn_info(self.molecule)
        self.assertEqual(len(mol_pairs[0]), 2)


if __name__ == "__main__":
    unittest.main()