import tensorflow as tf
import unittest
import os
import json
from megnet.data.molecule import SimpleMolGraph
from pymatgen import Molecule
import numpy as np

from megnet.data.molecule import MolecularGraph, mol_from_smiles, pybel

if pybel is None:
    import_failed = True
else:
    import_failed = False

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
        cls.mol = mol_from_smiles(cls.qm9_000001['smiles'])

    def setUp(self) -> None:
        self.mg = MolecularGraph()

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_featurizer(self):
        mg = MolecularGraph()
        mol_graph = mg.convert(self.mol)
        self.assertEqual(len(mol_graph['index1']), 20) # 20 bonds in total, including double counting
        self.assertEqual(len(mol_graph['atom']), 5) # 5 atoms
        self.assertListEqual(mol_graph['state'][0], [0, 0]) # dummy state [0, 0]
        mol_graph = mg.convert(self.mol, state_attributes=[[1, 2]])
        self.assertListEqual(mol_graph['state'][0], [1, 2])

    def test_simple_molecule_graph(self):
        mol = Molecule(['C', 'H', 'O'], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        graph = SimpleMolGraph().convert(mol)
        self.assertListEqual(graph['atom'], [6, 1, 8])
        self.assertTrue(np.allclose(graph['bond'], [1, 2, 1, 1, 2, 1]))
        self.assertListEqual(graph['index1'], [0, 0, 1, 1, 2, 2])
        self.assertListEqual(graph['index2'], [1, 2, 0, 2, 0, 1])

    def test_atom_features(self):
        """Detailed test of get_atom_feature"""

        # Test on Methane (atom 0 is an H)
        feat = self.mg.get_atom_feature(self.mol, self.mol.atoms[0])
        self.assertEqual(feat['element'], 'H')
        self.assertEqual(feat['atomic_num'], 1)
        self.assertEqual(feat['chirality'], 0)
        self.assertEqual(feat['formal_charge'], 0)
        self.assertEqual(feat['ring_sizes'], [])
        self.assertEqual(feat['hybridization'], 6)
        self.assertEqual(feat['acceptor'], False)
        self.assertEqual(feat['donor'], False)
        self.assertEqual(feat['aromatic'], False)

        # Test chirality using L/D-alanine
        la = pybel.readstring('smiles', 'N[C@@H](C)C(=O)O')
        feat = self.mg.get_atom_feature(la, la.atoms[1])
        self.assertEqual(feat['element'], 'C')
        self.assertEqual(feat['chirality'], 2)

        da = pybel.readstring('smiles', 'N[C@H](C)C(=O)O')
        feat = self.mg.get_atom_feature(da, da.atoms[1])
        self.assertEqual(feat['element'], 'C')
        self.assertEqual(feat['chirality'], 1)

        # Test formal charge
        proton = pybel.readstring('smiles', '[H+]')
        feat = self.mg.get_atom_feature(proton, proton.atoms[0])
        self.assertEqual(feat['element'], 'H')
        self.assertEqual(feat['formal_charge'], 1)

        # Test ring sizes
        naph = pybel.readstring('smiles', 'C1=CC=C2C=CC=CC2=C1')
        ring_sizes = [self.mg.get_atom_feature(naph, a)['ring_sizes'] for a in naph.atoms]
        self.assertEqual(ring_sizes.count([6]), 8)
        self.assertEqual(ring_sizes.count([6, 6]), 2)

        # Test aromicity
        aromicity = [self.mg.get_atom_feature(naph, a)['aromatic'] for a in naph.atoms]
        self.assertTrue(all(aromicity))

        # Test hydrogen bond acceptor
        ammonia = pybel.readstring('smiles', 'N')
        ammonia.addh()
        feat = self.mg.get_atom_feature(ammonia, ammonia.atoms[1])
        self.assertEqual(feat['element'], 'H')
        self.assertTrue(feat['donor'])
        self.assertFalse(feat['acceptor'])

        # Test hydrogen bond donor
        water = pybel.readstring('smiles', 'O')
        feat = self.mg.get_atom_feature(water, water.atoms[0])
        self.assertTrue(feat['acceptor'])


if __name__ == "__main__":
    unittest.main()
