"""
Temporary directory and file creation utilities.
"""

import os
import tempfile

from pathlib import Path

from monty.shutil import copy_r, remove


class ScratchDir:
    """
    .. note::

        With effect from Python 3.2, tempfile.TemporaryDirectory already
        implements much of the functionality of ScratchDir. However, it does
        not provide options for copying of files to and from (though it is
        possible to do this with other methods provided by shutil).

    Creates a "with" context manager that automatically handles creation of
    temporary directories (utilizing Python's build in temp directory
    functions) and cleanup when done. This improves on Python's built in
    functions by allowing for truly temporary workspace that are deleted
    when it is done. The way it works is as follows:

    1. Create a temp dir in specified root path.
    2. Optionally copy input files from current directory to temp dir.
    3. Change to temp dir.
    4. User performs specified operations.
    5. Optionally copy generated output files back to original directory.
    6. Change back to original directory.
    7. Delete temp dir.
    """

    SCR_LINK = "scratch_link"

    def __init__(
        self,
        rootpath,
        create_symbolic_link=False,
        copy_from_current_on_enter=False,
        copy_to_current_on_exit=False,
    ):
        """
        Initializes scratch directory given a **root** path. There is no need
        to try to create unique directory names. The code will generate a
        temporary sub directory in the rootpath. The way to use this is using a
        with context manager. Example::

            with ScratchDir("/scratch"):
                do_something()

        If the root path does not exist or is None, this will function as a
        simple pass through, i.e., nothing happens.

        Args:
            rootpath (str/Path): Path in which to create temp subdirectories.
                If this is None, no temp directories will be created and
                this will just be a simple pass through.
            create_symbolic_link (bool): Whether to create a symbolic link in
                the current working directory to the scratch directory
                created.
            copy_from_current_on_enter (bool): Whether to copy all files from
                the current directory (recursively) to the temp dir at the
                start, e.g., if input files are needed for performing some
                actions. Defaults to False.
            copy_to_current_on_exit (bool): Whether to copy files from the
                scratch to the current directory (recursively) at the end. E
                .g., if output files are generated during the operation.
                Defaults to False.
        """
        if Path is not None and isinstance(rootpath, Path):
            rootpath = str(rootpath)

        self.rootpath = os.path.abspath(rootpath) if rootpath is not None else None
        self.cwd = os.getcwd()
        self.create_symbolic_link = create_symbolic_link
        self.start_copy = copy_from_current_on_enter
        self.end_copy = copy_to_current_on_exit

    def __enter__(self):
        tempdir = self.cwd
        if self.rootpath is not None and os.path.exists(self.rootpath):
            tempdir = tempfile.mkdtemp(dir=self.rootpath)
            self.tempdir = os.path.abspath(tempdir)
            if self.start_copy:
                copy_r(".", tempdir)
            if self.create_symbolic_link:
                os.symlink(tempdir, ScratchDir.SCR_LINK)
            os.chdir(tempdir)
        return tempdir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rootpath is not None and os.path.exists(self.rootpath):
            if self.end_copy:
                files = set(os.listdir(self.tempdir))
                orig_files = set(os.listdir(self.cwd))

                # First copy files over
                copy_r(self.tempdir, self.cwd)

                # Delete any files that are now gone
                for f in orig_files - files:
                    fpath = os.path.join(self.cwd, f)
                    remove(fpath)

            os.chdir(self.cwd)
            remove(self.tempdir)
