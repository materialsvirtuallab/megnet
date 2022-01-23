import unittest
import os
import platform

from monty.os.path import which, zpath, find_exts
from monty.os import cd, makedirs_p

test_dir = os.path.join(os.path.dirname(__file__), "test_files")


class PathTest(unittest.TestCase):
    @unittest.skipIf(platform.system() == "Windows", "Skip on windows")
    def test_which(self):
        try:
            py = which("python3")
            self.assertEqual(os.path.basename(py), "python3")
        except:
            py = which("python")
            self.assertEqual(os.path.basename(py), "python")
        self.assertEqual("/usr/bin/find", which("/usr/bin/find"))
        self.assertIs(which("non_existent_exe"), None)

    def test_zpath(self):
        fullzpath = zpath(os.path.join(test_dir, "myfile_gz"))
        self.assertEqual(os.path.join(test_dir, "myfile_gz.gz"), fullzpath)

    def test_find_exts(self):
        self.assertTrue(len(find_exts(os.path.dirname(__file__), "py")) >= 18)
        self.assertEqual(len(find_exts(os.path.dirname(__file__), "bz2")), 2)
        self.assertEqual(
            len(find_exts(os.path.dirname(__file__), "bz2", exclude_dirs="test_files")),
            0,
        )
        self.assertEqual(
            len(find_exts(os.path.dirname(__file__), "bz2", include_dirs="test_files")),
            2,
        )


class CdTest(unittest.TestCase):
    def test_cd(self):
        with cd(test_dir):
            self.assertTrue(os.path.exists("empty_file.txt"))
        self.assertFalse(os.path.exists("empty_file.txt"))

    def test_cd_exception(self):
        with cd(test_dir):
            self.assertTrue(os.path.exists("empty_file.txt"))
        self.assertFalse(os.path.exists("empty_file.txt"))


class Makedirs_pTest(unittest.TestCase):
    def setUp(self):
        self.test_dir_path = os.path.join(test_dir, "test_dir")

    def test_makedirs_p(self):
        makedirs_p(self.test_dir_path)
        self.assertTrue(os.path.exists(self.test_dir_path))
        makedirs_p(self.test_dir_path)
        self.assertRaises(OSError, makedirs_p, os.path.join(test_dir, "myfile_txt"))

    def tearDown(self):
        os.rmdir(self.test_dir_path)


if __name__ == "__main__":
    unittest.main()
