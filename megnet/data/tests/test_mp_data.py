import unittest
from pymatgen.core import Structure
from megnet.data.mp import index_rep_from_structure, graph_to_inputs
from megnet.data.graph import GraphBatchDistanceConvert, GaussianDistance
import numpy as np
import os

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = [
            Structure.from_file(os.path.join(module_dir, "cifs", "LiFePO4_mp-19017_computed.cif")),
            Structure.from_file(os.path.join(module_dir, "cifs", "BaTiO3_mp-2998_computed.cif"))]

    def test_index_rep_from_structure(self):
        graph = index_rep_from_structure(self.structures[0])
        self.assertEqual(len(np.unique(graph['node'])), 4)
        self.assertEqual(len(np.unique(graph['index1'])),
                         len(self.structures[0]))
        self.assertEqual(len(np.unique(graph['index2'])),
                         len(self.structures[0]))
        self.assertEqual(len(graph['distance']), 704)

    def test_graph_to_inputs_and_class_generator(self):
        graphs = [index_rep_from_structure(i) for i in self.structures] * 4
        mp_ids = ['mp-19017', 'mp-2998'] * 5
        targets = [0.1, 0.2] * 5
        out = graph_to_inputs(mp_ids, graphs, targets)
        self.assertEqual(len(out), 7)
        gen = GraphBatchDistanceConvert(*out[:-1], batch_size=2, distance_convertor=GaussianDistance())
        data = gen[0]
        x = data[0]
        y = data[1]
        # only one graph, therefore the batch dimension is 1
        self.assertListEqual([i.shape[0] for i in x], [1] * len(x))
        # atom is 1*N where N is the total number of atoms
        self.assertEqual(len(x[0].shape), 2)
        # bond is 1*M*G
        self.assertEqual(len(x[1].shape), 3)
        # global is 1*U*K
        self.assertEqual(len(x[2].shape), 3)
        self.assertListEqual([len(x[i].shape) for i in range(3, 7)], [2] * 4)
        # target is 1*2(crystal)*1(target dimension)
        self.assertListEqual(list(y.shape), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
