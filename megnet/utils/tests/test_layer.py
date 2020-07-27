import tensorflow as tf
import unittest
from megnet.utils.layer import repeat_with_index, _repeat
import tensorflow as tf


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = tf.random.normal(shape=(1, 3, 4))
        cls.index = tf.convert_to_tensor(value=[0, 0, 0, 1, 1, 2])
        cls.n = tf.convert_to_tensor(value=[3, 2, 1])

    def test_repeat(self):
        repeat_result = _repeat(self.x, self.n, axis=1).numpy()
        self.assertListEqual(list(repeat_result.shape), [1, 6, 4])
        self.assertEqual(repeat_result[0, 0, 0], repeat_result[0, 1, 0])
        self.assertEqual(repeat_result[0, 0, 0], repeat_result[0, 2, 0])
        self.assertNotEqual(repeat_result[0, 0, 0], repeat_result[0, 3, 0])
        self.assertEqual(repeat_result[0, 3, 0], repeat_result[0, 4, 0])

    def test_repeat_with_index(self):
        repeat_result = repeat_with_index(self.x, self.index, axis=1).numpy()
        self.assertListEqual(list(repeat_result.shape), [1, 6, 4])
        self.assertEqual(repeat_result[0, 0, 0], repeat_result[0, 1, 0])
        self.assertEqual(repeat_result[0, 0, 0], repeat_result[0, 2, 0])
        self.assertNotEqual(repeat_result[0, 0, 0], repeat_result[0, 3, 0])
        self.assertEqual(repeat_result[0, 3, 0], repeat_result[0, 4, 0])


if __name__ == "__main__":
    unittest.main()
