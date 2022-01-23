import unittest

from monty.fnmatch import WildCard


class FuncTest(unittest.TestCase):
    def test_match(self):
        wc = WildCard("*.pdf")
        self.assertTrue(wc.match("A.pdf"))
        self.assertFalse(wc.match("A.pdg"))


if __name__ == "__main__":
    unittest.main()
