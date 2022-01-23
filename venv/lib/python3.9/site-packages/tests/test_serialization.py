import unittest
import os
import json
import glob

try:
    import msgpack
except ImportError:
    msgpack = None

from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir


class SerialTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Cleans up test files if a test fails
        files_to_clean_up = glob.glob("monte_test.*")
        for fn in files_to_clean_up:
            os.remove(fn)

    def test_dumpfn_loadfn(self):
        d = {"hello": "world"}

        # Test standard configuration
        for ext in (
            "json",
            "yaml",
            "yml",
            "json.gz",
            "yaml.gz",
            "json.bz2",
            "yaml.bz2",
        ):
            fn = f"monte_test.{ext}"
            dumpfn(d, fn)
            d2 = loadfn(fn)
            self.assertEqual(
                d,
                d2,
                msg=f"Test file with extension {ext} did not parse correctly",
            )
            os.remove(fn)

        # Test custom kwarg configuration
        dumpfn(d, "monte_test.json", indent=4)
        d2 = loadfn("monte_test.json")
        self.assertEqual(d, d2)
        os.remove("monte_test.json")
        dumpfn(d, "monte_test.yaml", default_flow_style=False)
        d2 = loadfn("monte_test.yaml")
        self.assertEqual(d, d2)
        os.remove("monte_test.yaml")

        # Check if fmt override works.
        dumpfn(d, "monte_test.json", fmt="yaml")
        with self.assertRaises(json.decoder.JSONDecodeError):
            d2 = loadfn("monte_test.json")
        d2 = loadfn("monte_test.json", fmt="yaml")
        self.assertEqual(d, d2)
        os.remove("monte_test.json")

        with self.assertRaises(TypeError):
            dumpfn(d, "monte_test.txt", fmt="garbage")
        with self.assertRaises(TypeError):
            loadfn("monte_test.txt", fmt="garbage")

    @unittest.skipIf(msgpack is None, "msgpack-python not installed.")
    def test_mpk(self):
        d = {"hello": "world"}

        # Test automatic format detection
        dumpfn(d, "monte_test.mpk")
        d2 = loadfn("monte_test.mpk")
        self.assertEqual(d, {k: v for k, v in d2.items()})
        os.remove("monte_test.mpk")

        # Test to ensure basename is respected, and not directory
        with ScratchDir("."):
            os.mkdir("mpk_test")
            os.chdir("mpk_test")
            fname = os.path.abspath("test_file.json")
            dumpfn({"test": 1}, fname)
            with open("test_file.json") as f:
                reloaded = json.loads(f.read())
            self.assertEqual(reloaded["test"], 1)


if __name__ == "__main__":
    unittest.main()
