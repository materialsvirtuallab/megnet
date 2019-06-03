from megnet.utils.metrics import mae, accuracy
import unittest
import numpy as np


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_true = np.array([0, 1, 2, 3, 4])
        cls.y_pred = np.array([0.1, 1.1, 2.1, 3.1, 4.1])

        cls.y_true_acc = np.array([0, 1, 1, 0])
        cls.y_pred_acc = np.array([0.1, 0.6, 0.4, 0.4])

    def test_mae(self):
        mae_result = mae(self.y_true, self.y_pred)
        self.assertAlmostEqual(mae_result, 0.1)

    def test_accuracy(self):
        acc = accuracy(self.y_true_acc, self.y_pred_acc)
        self.assertAlmostEqual(acc, 0.75)


if __name__ == "__main__":
    unittest.main()
