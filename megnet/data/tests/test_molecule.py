import tensorflow as tf
import unittest
import os
import json

from megnet.data.molecule import SimpleMolGraph
from megnet.data.graph import DummyConverter
from megnet.utils.general import to_list
from pymatgen import Molecule
import numpy as np

from megnet.data.molecule import MolecularGraph, MolecularGraphBatchGenerator,\
    pybel, mol_from_smiles, ring_to_vector

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
    def test_simple_molecule_graph(self):
        mol = Molecule(['C', 'H', 'O'], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        graph = SimpleMolGraph().convert(mol)
        self.assertListEqual(to_list(graph['atom']), [6, 1, 8])
        self.assertTrue(np.allclose(graph['bond'], [1, 2, 1, 1, 2, 1]))
        self.assertListEqual(to_list(graph['index1']), [0, 0, 1, 1, 2, 2])
        self.assertListEqual(to_list(graph['index2']), [1, 2, 0, 2, 0, 1])

    def test_ring_to_vector(self):
        x = [2, 2, 3]
        expected = [0, 2, 1, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected, ring_to_vector(x))


class MolecularGraphTest(unittest.TestCase):

    @classmethod
    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def setUpClass(cls):
        with open(os.path.join(module_dir, 'qm9', '000001.json'), 'r') as f:
            cls.qm9_000001 = json.load(f)
        cls.mol = mol_from_smiles(cls.qm9_000001['smiles'])

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def setUp(self) -> None:
        self.mg = MolecularGraph()

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_featurizer(self):
        mg = MolecularGraph()
        mol_graph = mg.convert(self.mol)
        self.assertEqual(len(mol_graph['index1']), 20)  # 20 bonds, including double counting
        self.assertEqual(len(mol_graph['atom']), 5)  # 5 atoms
        self.assertAlmostEqual(mol_graph['state'][0][0], 3.2, places=1)
        self.assertAlmostEqual(mol_graph['state'][0][1], 0.8, places=1)
        mol_graph = mg.convert(self.mol, state_attributes=[[1, 2]])
        self.assertListEqual(mol_graph['state'][0], [1, 2])

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
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

        # Make sure it gets the hybridization of the C correctly
        feat = self.mg.get_atom_feature(self.mol, self.mol.atoms[1])
        self.assertEqual(feat['element'], 'C')
        self.assertEqual(feat['atomic_num'], 6)
        self.assertEqual(feat['chirality'], 0)
        self.assertEqual(feat['formal_charge'], 0)
        self.assertEqual(feat['ring_sizes'], [])
        self.assertEqual(feat['hybridization'], 3)

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

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_atom_feature_vector(self):
        """Test the code that transforms feature dict to a list"""

        # Make feature dictionary with complicated molecule
        naph = pybel.readstring('smiles', 'C1=CC=C2C=CC=CC2=C1')
        feat = self.mg.get_atom_feature(naph, naph.atoms[3])

        # Run with the default features
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual(27, len(vec))

        # Check the on-hot-encoding for elements
        self.mg.atom_features = ['element']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([0, 1, 0, 0, 0], vec)

        # Check with only atomic number and formal charge
        self.mg.atom_features = ['atomic_num', 'formal_charge']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([6, 0], vec)

        # Make sure it obeys user-defined order
        self.mg.atom_features = ['formal_charge', 'atomic_num']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([0, 6], vec)

        # Run the chirality binarization
        self.mg.atom_features = ['chirality']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([1, 0, 0], vec)

        # Run the ring size calculation (it is in 2 6-member rings)
        self.mg.atom_features = ['ring_sizes']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([0, 0, 0, 0, 0, 2, 0, 0, 0], vec)

        # Run the hybridization test
        self.mg.atom_features = ['hybridization']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([0, 1, 0, 0, 0, 0], vec)

        # Test donor, acceptor, aromatic
        self.mg.atom_features = ['donor', 'acceptor', 'aromatic']
        vec = self.mg._create_atom_feature_vector(feat)
        self.assertEqual([0, 0, 1], vec)

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_bond_features(self):
        """Detailed tests for bond features"""

        # Test C-H bonds on the methane molecule
        feat = self.mg.get_pair_feature(self.mol, 0, 1, True)
        self.assertEqual(0, feat['a_idx'])
        self.assertEqual(1, feat['b_idx'])
        self.assertEqual(1, feat['bond_type'])
        self.assertEqual(False, feat['same_ring'])
        self.assertAlmostEqual(1.0921, feat['spatial_distance'], places=3)

        feat = self.mg.get_pair_feature(self.mol, 1, 0, True)
        self.assertEqual(1, feat['a_idx'])
        self.assertEqual(0, feat['b_idx'])
        self.assertEqual(1, feat['bond_type'])
        self.assertEqual(False, feat['same_ring'])
        self.assertAlmostEqual(1.0921, feat['spatial_distance'], places=3)

        # Test atoms that are not bonded
        feat = self.mg.get_pair_feature(self.mol, 0, 2, True)
        self.assertEqual(0, feat['a_idx'])
        self.assertEqual(2, feat['b_idx'])
        self.assertEqual(0, feat['bond_type'])
        self.assertEqual(False, feat['same_ring'])
        self.assertAlmostEqual(1.7835, feat['spatial_distance'], places=3)

        feat = self.mg.get_pair_feature(self.mol, 0, 2, False)
        self.assertIsNone(feat)

        # Test an aromatic bond
        benzene = pybel.readstring('smiles', 'C1=CC=CC=C1')
        feat = self.mg.get_pair_feature(benzene, 0, 1, True)
        self.assertEqual(4, feat['bond_type'])
        self.assertEqual(True, feat['same_ring'])

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_bond_feature_vec(self):
        # Test the full list
        feat = self.mg.get_pair_feature(self.mol, 0, 1, True)
        self.assertEqual(26, len(self.mg._create_pair_feature_vector(feat)))

        # Test the bond type
        self.mg.bond_features = ['bond_type']
        self.assertEqual([0, 1, 0, 0, 0], self.mg._create_pair_feature_vector(feat))

        # Test the ring encoding
        self.mg.bond_features = ['same_ring']
        self.assertEqual([0], self.mg._create_pair_feature_vector(feat))

        # Test the spatial distance
        self.mg.bond_features = ['spatial_distance']
        self.assertEqual(20, len(self.mg._create_pair_feature_vector(feat)))

        # Test the spatial distance without the expansion
        self.mg.distance_converter = DummyConverter()
        self.assertAlmostEqual(1.0921, self.mg._create_pair_feature_vector(feat)[0], places=3)

    @unittest.skipIf(import_failed, "molecule package relies on openbabel")
    def test_mol_generator(self):
        mols = ['c', 'C', 'cc', 'ccn']
        gen = MolecularGraphBatchGenerator(mols, range(4), batch_size=2, molecule_format='smiles')

        # Make a batch, check it has the correct sizes
        batch = gen[0]
        self.assertEqual(2, len(batch))
        self.assertEqual((1, 1, 2), np.shape(batch[1]))  # Should be 2 targets
        self.assertEqual(7, len(batch[0]))  # Should have 7 different arrays for inputs

        # Test the generator with 2 threads
        gen = MolecularGraphBatchGenerator(mols, range(4), batch_size=2,
                                           molecule_format='smiles', n_jobs=2)
        batch = gen[0]
        self.assertEqual(2, len(batch))

        # Create the cached generator, amke sure it creates properly-sized inputs
        cached = gen.create_cached_generator()

        batch = cached[0]
        self.assertEqual(2, len(batch))
        self.assertEqual(2, np.size(batch[1]))  # Should be 2 targets
        self.assertEqual(7, len(batch[0]))  # Should have 7 different arrays for inputs


if __name__ == "__main__":
    unittest.main()
