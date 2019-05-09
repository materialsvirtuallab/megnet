from megnet.utils.preprocessing import StandardScaler
import unittest


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Since only len(structure) is called, we will
        # use a dummy list as structure
        cls.structures = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
        cls.targets = [2, 3, 4]

    def test_from_training(self):
        scaler = StandardScaler.from_training_data(self.structures, self.targets, is_intensive=False)
        self.assertEqual(scaler.mean, 1)
        self.assertEqual(scaler.std, 1)

    def test_transform_inverse_transform(self):
        scaler = StandardScaler.from_training_data(self.structures, self.targets, is_intensive=False)
        transformed_target = scaler.transform(100, 1)
        orig_target = scaler.inverse_transform(transformed_target, 1)
        self.assertAlmostEqual(100, orig_target)
        scaler = StandardScaler.from_training_data(self.structures, self.targets, is_intensive=True)
        transformed_target = scaler.transform(100, 1)
        orig_target = scaler.inverse_transform(transformed_target, 1)
        self.assertAlmostEqual(100, orig_target)


if __name__ == "__main__":
    unittest.main()

