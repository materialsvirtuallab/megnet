import unittest
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph, get_elemental_embeddings
from pymatgen import Structure
import os
import numpy as np

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = [
            Structure.from_file(os.path.join(module_dir, "cifs", "LiFePO4_mp-19017_computed.cif")),
            Structure.from_file(os.path.join(module_dir, "cifs", "BaTiO3_mp-2998_computed.cif"))]

    def test_crystalgraph(self):
        cg = CrystalGraph(cutoff=4)
        graph = cg.convert(self.structures[0])
        self.assertEqual(cg.cutoff, 4)
        keys = set(graph.keys())
        self.assertSetEqual({"bond", "atom", "index1", "index2", "state"}, keys)
        cg2 = CrystalGraph(cutoff=6)
        self.assertEqual(cg2.cutoff, 6)
        graph2 = cg2.convert(self.structures[0])
        self.assertListEqual(graph2['state'][0], [0, 0])
        graph3 = cg(self.structures[0])
        self.assertListEqual(graph['atom'], graph3['atom'])

    def test_convert(self):
        cg = CrystalGraph(cutoff=4)
        graph = cg.convert(self.structures[0])
        self.assertListEqual(graph['atom'], [i.specie.Z for i in self.structures[0]])

    def test_get_input(self):
        cg = CrystalGraph(cutoff=4, bond_convertor=GaussianDistance(np.linspace(0, 5, 100), 0.5))
        inp = cg.get_input(self.structures[0])
        self.assertEqual(len(inp), 7)
        shapes = [i.shape for i in inp]
        true_shapes = [(1, 28), (1, 704, 100), (1, 1, 2), (1, 704), (1, 704), (1, 28), (1, 704)]
        for i, j in zip(shapes, true_shapes):
            self.assertListEqual(list(i), list(j))

    def test_get_flat_data(self):
        cg = CrystalGraph(cutoff=4)
        graphs = [cg.convert(i) for i in self.structures]
        targets = [0.1, 0.2]
        inp = cg.get_flat_data(graphs, targets)
        self.assertListEqual([len(i) for i in inp], [2] * 6)

    def test_get_elemental_embeddings(self):
        data = get_elemental_embeddings()
        for k, v in data.items():
            self.assertTrue(len(v) == 16)


if __name__ == "__main__":
    unittest.main()
