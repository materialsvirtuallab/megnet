import unittest
import os
from monty.re import regrep

test_dir = os.path.join(os.path.dirname(__file__), "test_files")


class RegrepTest(unittest.TestCase):
    def test_regrep(self):
        """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
        fname = os.path.join(test_dir, "3000_lines.txt")
        matches = regrep(fname, {"1": r"1(\d+)", "3": r"3(\d+)"}, postprocess=int)
        self.assertEqual(len(matches["1"]), 1380)
        self.assertEqual(len(matches["3"]), 571)
        self.assertEqual(matches["1"][0][0][0], 0)

        matches = regrep(
            fname,
            {"1": r"1(\d+)", "3": r"3(\d+)"},
            reverse=True,
            terminate_on_match=True,
            postprocess=int,
        )
        self.assertEqual(len(matches["1"]), 1)
        self.assertEqual(len(matches["3"]), 11)


if __name__ == "__main__":
    unittest.main()
