import unittest

import numpy as np

from megnet.data.graph import (
    EmbeddingMap,
    GaussianDistance,
    GraphBatchDistanceConvert,
    GraphBatchGenerator,
)


class TestGraph(unittest.TestCase):
    def test_gaussian_expansion(self):
        x = np.random.normal(size=(10,))
        centers = np.linspace(0, 5, 20)
        width = 0.5
        gd = GaussianDistance(centers, width)
        out = gd.convert(x)
        self.assertListEqual(list(out.shape), [10, 20])

    def test_graph_generator(self):
        feature = [np.random.normal(size=(3, 4)), np.random.normal(size=(2, 4))]
        bond = [np.random.normal(size=(2, 5)), np.random.normal(size=(1, 5))]
        glob_features = [np.random.normal(size=(1, 2)), np.random.normal(size=(1, 2))]
        index1 = [np.array([0, 1]), np.array([0])]
        index2 = [np.array([1, 2]), np.array([1])]
        targets = np.random.normal(size=(2, 1))
        gen = GraphBatchGenerator(feature, bond, glob_features, index1, index2, targets, batch_size=2)
        data = gen[0]
        self.assertListEqual(list(data[0][0].shape), [1, 5, 4])
        self.assertListEqual(list(data[0][1].shape), [1, 3, 5])
        self.assertListEqual(list(data[0][2].shape), [1, 2, 2])
        self.assertListEqual(list(data[0][3].shape), [1, 3])
        self.assertListEqual(list(data[0][4].shape), [1, 3])
        self.assertListEqual(list(data[1].shape), [1, 2, 1])

        # Make sure it still functions if a target is not provided
        gen = GraphBatchGenerator(feature, bond, glob_features, index1, index2, batch_size=2)

        data = gen[0]
        self.assertEqual(7, len(data))  # Should only be the inputs
        self.assertListEqual(list(data[0].shape), [1, 5, 4])

        # when bonds are one dimension arrays

        bond = [np.random.normal(size=(2,)), np.random.normal(size=(1,))]
        gen = GraphBatchGenerator(feature, bond, glob_features, index1, index2, targets, batch_size=2)
        data = gen[0]
        self.assertListEqual(list(data[0][1].shape), [1, 3])

    def test_graph_batch_distance_converter(self):
        feature = [np.random.normal(size=(3, 4)), np.random.normal(size=(2, 4))]
        bond = [np.random.normal(size=(2,)), np.random.normal(size=(1,))]
        glob_features = [np.random.normal(size=(1, 2)), np.random.normal(size=(1, 2))]
        index1 = [np.array([0, 1]), np.array([0])]
        index2 = [np.array([1, 2]), np.array([1])]
        targets = np.random.normal(size=(2, 1))
        centers = np.linspace(0, 5, 20)
        width = 0.5
        gen = GraphBatchDistanceConvert(
            feature,
            bond,
            glob_features,
            index1,
            index2,
            targets,
            batch_size=2,
            distance_converter=GaussianDistance(centers, width),
        )
        data = gen[0]
        self.assertListEqual(list(data[0][0].shape), [1, 5, 4])
        self.assertListEqual(list(data[0][1].shape), [1, 3, 20])
        self.assertListEqual(list(data[0][2].shape), [1, 2, 2])
        self.assertListEqual(list(data[0][3].shape), [1, 3])
        self.assertListEqual(list(data[0][4].shape), [1, 3])
        self.assertListEqual(list(data[1].shape), [1, 2, 1])

    def test_embedding_map(self):
        m = EmbeddingMap(np.array([[1, 2], [3, 4]]))
        res = m.convert(np.array([0, 0, 1, 1]))
        self.assertListEqual(res[0].tolist(), [1, 2])
        self.assertListEqual(res[2].tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
