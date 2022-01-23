import unittest
import shutil
import os

from monty.tempfile import ScratchDir

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class ScratchDirTest(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        os.chdir(test_dir)
        self.scratch_root = os.path.join(test_dir, "..", "..", "tempscratch")
        os.mkdir(self.scratch_root)

    def test_with_copy(self):
        # We write a pre-scratch file.
        with open("pre_scratch_text", "w") as f:
            f.write("write")

        with ScratchDir(
            self.scratch_root,
            copy_from_current_on_enter=True,
            copy_to_current_on_exit=True,
        ) as d:
            with open("scratch_text", "w") as f:
                f.write("write")
            files = os.listdir(d)
            self.assertIn("scratch_text", files)
            self.assertIn("empty_file.txt", files)
            self.assertIn("pre_scratch_text", files)

            # We remove the pre-scratch file.
            os.remove("pre_scratch_text")

        # Make sure the tempdir is deleted.
        self.assertFalse(os.path.exists(d))
        files = os.listdir(".")
        self.assertIn("scratch_text", files)

        # We check that the pre-scratch file no longer exists (because it is
        # deleted in the scratch)
        self.assertNotIn("pre_scratch_text", files)
        os.remove("scratch_text")

    def test_no_copy(self):

        with ScratchDir(
            self.scratch_root,
            copy_from_current_on_enter=False,
            copy_to_current_on_exit=False,
        ) as d:
            with open("scratch_text", "w") as f:
                f.write("write")
            files = os.listdir(d)
            self.assertIn("scratch_text", files)
            self.assertNotIn("empty_file.txt", files)

        # Make sure the tempdir is deleted.
        self.assertFalse(os.path.exists(d))
        files = os.listdir(".")
        self.assertNotIn("scratch_text", files)

    def test_bad_root(self):
        with ScratchDir("bad_groot") as d:
            self.assertEqual(d, test_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.scratch_root)


if __name__ == "__main__":
    unittest.main()
