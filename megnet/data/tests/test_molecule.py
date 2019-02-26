import unittest
import os
import json
try:
    from megnet.data.molecule import MolecularGraph, mol_from_smiles
    import_failed = False
except ImportError:
    import_failed = True

module_dir = os.path.dirname(os.path.abspath(__file__))


def equal(x, y):
    if isinstance(x, list):
        return all([i == j for i, j in zip(x, y)])
    if isinstance(x, float):
        return abs(x-y) < 0.01
    else:
        return x == y


def is_subset(xs, ys, sort_attr='atomic_num', skip=None):
    xs = sorted(xs, key=lambda x: x[sort_attr])
    ys = sorted(ys, key=lambda x: x[sort_attr])
    for x, y in zip(xs, ys):
        for i in x:
            if skip:
                if i in skip:
                    continue
            if equal(y[i], x[i]):
                pass
            else:
                print(i, x[i], y[i])
                return False
    return True


class QM9Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(module_dir, 'qm9', '000001.json'), 'r') as f:
            cls.qm9_000001 = json.load(f)

    @unittest.skipIf(import_failed, "molecular package replies on openbabel")
    def test_featurizer(self):
        mg = MolecularGraph()
        mol = mol_from_smiles(self.qm9_000001['smiles'])
        atom_attributes, bond_attributes = mg.featurize(mol)
        self.assertTrue(is_subset(atom_attributes, self.qm9_000001['atoms']))
        self.assertTrue(is_subset(bond_attributes, self.qm9_000001['atom_pairs'], 'spatial_distance', skip=['a_idx', 'b_idx']))


if __name__ == "__main__":
    unittest.main()
