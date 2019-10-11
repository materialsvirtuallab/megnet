import unittest
from pymatgen import Structure, Lattice
from megnet.utils.data import get_graphs_within_cutoff


class TestGeneralUtils(unittest.TestCase):
    def test_model_load(self):
        s = Structure(Lattice.cubic(3.6), ['Mo', 'Mo'], [[0.5, 0.5, 0.5], [0, 0, 0]])
        center_indices, neighbor_indices, images, distances = \
            get_graphs_within_cutoff(s, 4)
        self.assertListEqual(center_indices.tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

