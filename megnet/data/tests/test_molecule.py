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


class QM9Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(module_dir, 'qm9', '000001.json'), 'r') as f:
            cls.qm9_000001 = json.load(f)

    @unittest.skipIf(import_failed, "molecular package replies on openbabel")
    def test_featurizer(self):
        mg = MolecularGraph()
        mol = mol_from_smiles(self.qm9_000001['smiles'])
        mol_graph = mg.convert(mol)
        self.assertEqual(len(mol_graph['index1']), 20) # 20 bonds in total, including double counting
        self.assertEqual(len(mol_graph['atom']), 5) # 5 atoms
        self.assertListEqual(mol_graph['state'][0], [0, 0]) # dummy state [0, 0]
        mol_graph = mg.convert(mol, state_attributes=[[1, 2]])
        self.assertListEqual(mol_graph['state'][0], [1, 2])


if __name__ == "__main__":
    unittest.main()
