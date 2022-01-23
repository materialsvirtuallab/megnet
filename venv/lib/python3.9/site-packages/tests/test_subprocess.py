import unittest

from monty.subprocess import Command


class CommandTest(unittest.TestCase):
    def test_command(self):
        """Test Command class"""
        sleep05 = Command("sleep 0.5")

        sleep05.run(timeout=1)
        print(sleep05)
        self.assertEqual(sleep05.retcode, 0)
        self.assertFalse(sleep05.killed)

        sleep05.run(timeout=0.1)
        self.assertNotEqual(sleep05.retcode, 0)
        self.assertTrue(sleep05.killed)


if __name__ == "__main__":
    unittest.main()
