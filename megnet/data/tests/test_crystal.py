import os
import unittest

import numpy as np
from pymatgen.core import Structure

from megnet.data.crystal import (
    CrystalGraph,
    CrystalGraphDisordered,
    CrystalGraphWithBondTypes,
    get_elemental_embeddings,
)
from megnet.data.graph import GaussianDistance
from megnet.utils.general import to_list

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = [
            Structure.from_file(os.path.join(module_dir, "cifs", "LiFePO4_mp-19017_computed.cif")),
            Structure.from_file(os.path.join(module_dir, "cifs", "BaTiO3_mp-2998_computed.cif")),
        ]

    def test_crystalgraph(self):
        cg = CrystalGraph(cutoff=4)
        graph = cg.convert(self.structures[0])
        self.assertEqual(cg.cutoff, 4)
        keys = set(graph.keys())
        self.assertSetEqual({"bond", "atom", "index1", "index2", "state"}, keys)
        cg2 = CrystalGraph(cutoff=6)
        self.assertEqual(cg2.cutoff, 6)
        graph2 = cg2.convert(self.structures[0])
        self.assertListEqual(to_list(graph2["state"][0]), [0, 0])
        graph3 = cg(self.structures[0])
        np.testing.assert_almost_equal(graph["atom"], graph3["atom"])

    def test_crystalgraph_disordered(self):
        cg = CrystalGraphDisordered(cutoff=4.0)
        graph = cg.convert(self.structures[0])
        self.assertEqual(cg.atom_converter.convert(graph["atom"]).shape[1], 16)

    def test_crystal_graph_with_bond_types(self):
        graph = {
            "atom": [11, 8, 8],
            "index1": [0, 0, 1, 1, 2, 2],
            "index2": [0, 1, 2, 2, 1, 1],
            "bond": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "state": [[0, 0]],
        }
        cgbt = CrystalGraphWithBondTypes(nn_strategy="VoronoiNN")
        new_graph = cgbt._get_bond_type(graph)
        self.assertListEqual(to_list(new_graph["bond"]), [2, 1, 0, 0, 0, 0])

    def test_convert(self):
        cg = CrystalGraph(cutoff=4)
        graph = cg.convert(self.structures[0])
        self.assertListEqual(to_list(graph["atom"]), [i.specie.Z for i in self.structures[0]])

    def test_get_input(self):
        cg = CrystalGraph(cutoff=4, bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5))
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
